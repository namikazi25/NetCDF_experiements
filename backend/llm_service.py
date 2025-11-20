import os
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

# Default to OpenRouter, but allow override
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower() # auto, openrouter, local

def get_client():
    if LLM_PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError("LLM_PROVIDER is 'openrouter' but OPENROUTER_API_KEY is not set.")
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY), os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
    
    elif LLM_PROVIDER == "local":
        return OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio"), "local-model"
    
    else: # "auto"
        if OPENROUTER_API_KEY:
            return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY), os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
        else:
            return OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio"), "local-model"

client, MODEL = get_client()

def format_metadata_context(metadata: dict) -> str:
    # Check if this is a multi-file dict
    if "files" in metadata:
        context = "NetCDF Files Metadata:\n"
        for filename, file_meta in metadata["files"].items():
            context += f"\n--- File: {filename} ---\n"
            context += f"Global Attributes: {json.dumps(file_meta.get('attrs', {}), indent=2)}\n"
            context += f"Dimensions: {json.dumps(file_meta.get('dims', {}), indent=2)}\n"
            context += f"Variables: {json.dumps(file_meta.get('data_vars', {}), indent=2)}\n"
        return context
    else:
        # Single file fallback
        return f"""
        NetCDF File Metadata:
        Global Attributes: {json.dumps(metadata.get('attrs', {}), indent=2)}
        Dimensions: {json.dumps(metadata.get('dims', {}), indent=2)}
        Variables: {json.dumps(metadata.get('data_vars', {}), indent=2)}
        """

def generate_suggestions(metadata: dict) -> dict:
    """
    Generates a summary and 3 suggested queries based on the metadata.
    Returns a JSON-compatible dict with 'summary' and 'suggestions' (list of strings).
    """
    context = format_metadata_context(metadata)
    
    system_prompt = """You are a helpful data scientist. 
    Analyze the provided NetCDF metadata.
    1. Provide a brief, 2-sentence summary of the dataset (what variables, dimensions, and context it covers).
    2. Generate 3 specific, interesting questions a user could ask about this data to get insights.
    
    IMPORTANT: The user has a Python environment. Suggest questions that can be answered by calculating numbers or plotting data.
    Examples:
    - "Plot the average surface elevation map."
    - "Calculate the maximum depth in the region."
    - "Show the temperature profile at node 100."
    
    Output strictly in JSON format:
    {
        "summary": "...",
        "suggestions": ["Question 1", "Question 2", "Question 3"]
    }
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the metadata:\n{context}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        content = response.choices[0].message.content
        
        # Clean up content if it has markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return json.loads(content)
    except Exception as e:
        # Fallback if JSON parsing fails or API error
        return {
            "summary": "Could not generate analysis.",
            "suggestions": ["What variables are in this file?", "Show me the dimensions.", "Describe the attributes."]
        }

def chat_with_context(query: str, metadata: dict) -> str:
    context = format_metadata_context(metadata)
    
    system_prompt = """You are a helpful assistant that analyzes NetCDF file metadata. 
    User will provide metadata about a scientific dataset. 
    Answer their questions based on this metadata. 
    If asked to write code to load this data, assume 'xarray' is available."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the file metadata:\n{context}\n\nUser Query: {query}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with LLM: {str(e)}"
