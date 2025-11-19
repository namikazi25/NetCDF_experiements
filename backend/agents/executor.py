import json
from llm_service import client, MODEL
from code_executor import execute_python_code

def generate_and_execute_code(query: str, plan: dict, netcdf_path: str, scenario_path: str = None) -> dict:
    """
    Generates Python code based on the approved plan and executes it.
    """
    system_prompt = """You are a Python Code Generator.
    Your task is to write Python code to execute the provided PLAN.
    
    CONTEXT:
    - Baseline File: `netcdf_path` (Variable: `ds_base` or `ds`)
    - Scenario File: `scenario_path` (Variable: `ds_comp` - ONLY if provided)
    
    SETUP INSTRUCTIONS:
    - ALWAYS load the baseline: `ds = xr.open_dataset(netcdf_path)` (alias `ds_base = ds`)
    - IF `scenario_path` is not None:
      `ds_comp = xr.open_dataset(scenario_path)`
      `print("Comparison mode: ds_base vs ds_comp")`
      
    - Libraries: xarray (xr), numpy (np), matplotlib.pyplot (plt), scipy
    - Helper: `plot_unstructured(variable, x, y)` for SCHISM grids.
    
    RULES:
    1. Follow the plan steps exactly.
    2. Use `print()` for text results.
    3. Use `plt.show()` or `plt.savefig()` for plots.
    4. Output ONLY valid Python code. No markdown blocks if possible, but I will strip them.
    """
    
    plan_str = "\n".join(plan.get("steps", []))
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Plan:\n{plan_str}\n\nWrite the code."}
    ]
    
    max_retries = 3
    current_code = None
    
    # Initial generation
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        current_code = response.choices[0].message.content
    except Exception as e:
        return {"success": False, "stderr": f"Initial Code Gen Error: {e}", "stdout": "", "images": []}

    for attempt in range(max_retries):
        # Clean code
        if "```python" in current_code:
            code_to_run = current_code.split("```python")[1].split("```")[0].strip()
        elif "```" in current_code:
            code_to_run = current_code.split("```")[1].split("```")[0].strip()
        else:
            code_to_run = current_code.strip()
            
        # Execute
        result = execute_python_code(code_to_run, netcdf_path, scenario_path)
        
        # If successful or no stderr, return result
        if result["success"] and not result["stderr"]:
            result["code_generated"] = code_to_run
            return result
            
        # If failed, try to fix
        error_msg = result["stderr"]
        print(f"Attempt {attempt+1} failed: {error_msg}")
        
        if attempt < max_retries - 1:
            fix_prompt = f"""The code failed with this error:
            {error_msg}
            
            Analyze why, fix the code, and output the FULL corrected code block.
            """
            
            messages.append({"role": "assistant", "content": current_code})
            messages.append({"role": "user", "content": fix_prompt})
            
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages
                )
                current_code = response.choices[0].message.content
            except Exception as e:
                return {"success": False, "stderr": f"Fix Gen Error: {e}", "stdout": "", "images": []}
        else:
            # Out of retries, return the last failed result
            result["code_generated"] = code_to_run
            return result

    return {"success": False, "stderr": "Max retries exceeded", "stdout": "", "images": []}
