import json
from llm_service import client, MODEL, format_metadata_context

def plan_task(query: str, metadata: dict) -> dict:
    """
    Decomposes the user query into a list of steps.
    Returns a JSON dict with 'thought' and 'steps'.
    """
    context = format_metadata_context(metadata)
    
    system_prompt = """You are a Senior Data Scientist acting as a Planner.
    Your goal is to break down a user's request into logical steps for a Python Executor.
    
    Available Tools:
    - Python Code Execution (xarray, numpy, matplotlib, scipy)
    - NetCDF Metadata Access (Provided in context)
    
    SCHISM VARIABLE MAP (Use these standard names if present):
    - Surface Elevation: `elev` (NOT `zcor`)
    - Water Depth: `depth`
    - Velocity: `hvel_x`, `hvel_y` (Horizontal), `vertical_velocity` (Vertical)
    - Salinity: `salt`
    - Temperature: `temp`
    
    COMPARISON MODE RULES (Triggered by words like "difference", "change", "impact"):
    1. Assume TWO datasets are loaded: `ds_base` (Baseline) and `ds_comp` (Scenario).
    2. Step 1 should be: "Ensure both `ds_base` and `ds_comp` are loaded."
    3. Calculate difference: `diff = ds_comp['variable'] - ds_base['variable']`.
    4. Use `diff` for plotting or statistics.
    
    CRITICAL RULES:
    1. DO NOT ask to print `ds.info()` or `ds.variables`. The metadata is already provided.
    2. Trust the metadata provided in the context.
    3. If the user asks for "elevation", use `elev`. `zcor` is for 3D vertical coordinates.
    4. SCHISM Support:
       - If the user asks for a map/plot of unstructured data, specify using `plot_unstructured(variable, x, y)`.
    
    OUTPUT FORMAT:
    {
        "thought": "Reasoning about the request. I see variable 'elev' in the metadata...",
        "steps": [
            "Step 1: Load the dataset...",
            "Step 2: Extract variable 'elev'...",
            "Step 3: Calculate..."
        ]
    }
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Metadata:\n{context}\n\nUser Query: {query}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        content = response.choices[0].message.content
        
        # JSON Cleanup
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return json.loads(content)
    except Exception as e:
        return {"thought": f"Error planning: {e}", "steps": []}
