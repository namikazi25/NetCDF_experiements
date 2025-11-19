import json
from llm_service import client, MODEL, format_metadata_context
from code_executor import execute_python_code

def run_agent_workflow(query: str, metadata: dict, netcdf_path: str) -> dict:
    """
    Executes a ReAct loop to answer the user's query using Python code.
    Returns a dict with 'response' (text) and 'images' (list of base64).
    """
    context = format_metadata_context(metadata)
    
    system_prompt = """You are a Python Data Scientist.
    You have access to a Python environment with xarray, numpy, scipy, and matplotlib.
    
    Your goal is to answer the user's question by writing Python code.
    
    RULES:
    1. Do NOT describe how to solve the problem. Write the code to solve it.
    2. Use `print()` to output specific numbers or findings.
    3. Use `plt.show()` or `plt.savefig()` to generate plots.
    4. For unstructured grids (SCHISM), use the helper function:
       `plot_unstructured(variable, node_x, node_y, title="...")`
       where `variable` is the data at nodes, and `node_x`, `node_y` are coordinates.
    5. The NetCDF file path is available as the variable `netcdf_path`.
    6. Open the file using `ds = xr.open_dataset(netcdf_path)`.
    
    RESPONSE FORMAT:
    You must output a JSON object with the following structure:
    {
        "thought": "Brief reasoning about what code to write.",
        "code": "The python code to execute."
    }
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Metadata:\n{context}\n\nUser Query: {query}"}
    ]
    
    # Step 1: Generate Code
    try:
        # Note: response_format={"type": "json_object"} is not supported by all local models/servers
        # We will rely on the prompt to enforce JSON
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
            
        # Try to parse JSON
        try:
            plan = json.loads(content)
        except json.JSONDecodeError:
            # If simple parsing fails, try to find the first { and last }
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                plan = json.loads(content[start:end+1])
            else:
                raise Exception("Could not parse JSON plan from LLM response.")
        
        code = plan.get("code")
        thought = plan.get("thought")
        
        if not code:
            return {"response": f"I couldn't generate code to answer that. Thought: {thought}", "images": []}
            
    except Exception as e:
        error_msg = str(e)
        if "Connection error" in error_msg or "connection" in error_msg.lower():
            return {
                "response": f"**Connection Error**: {error_msg}", 
                "images": []
            }
        # If we failed to parse the plan, return the raw content for debugging
        return {"response": f"Error parsing plan: {error_msg}\n\nRaw Output:\n{content}", "images": []}
        
    # Step 2: Execute Code
    exec_result = execute_python_code(code, netcdf_path)
    
    # Step 3: Synthesize Answer
    if exec_result["success"]:
        final_prompt = f"""
        User Query: {query}
        
        My Plan: {thought}
        
        Code Execution Output:
        {exec_result['stdout']}
        
        Code Execution Errors (if any):
        {exec_result['stderr']}
        
        Based on the code output, provide a concise final answer to the user. 
        If there are plots, mention them.
        """
        
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": final_prompt})
        
        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        
        return {
            "response": final_response.choices[0].message.content,
            "images": exec_result["images"]
        }
    else:
        # Error handling / Self-Correction could go here
        return {
            "response": f"I tried to run some code but it failed.\n\nError:\n```\n{exec_result['stderr']}\n```",
            "images": []
        }
