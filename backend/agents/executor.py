import json
from llm_service import client, MODEL
from code_executor import execute_python_code

def generate_and_execute_code(query: str, plan: dict, netcdf_path: str, scenario_path: str = None) -> dict:
    """
    Generates Python code based on the approved plan and executes it.
    """
    system_prompt = """You are a Python Code Generator.
    Your task is to write Python code to execute the provided PLAN.
    
    CONTEXT VARIABLES (ALREADY LOADED):
    - `netcdf_path`: Path to the Baseline file.
    - `scenario_path`: Path to the Scenario file.
    
    AVAILABLE TOOLS (INJECTED):
    - `get_schism_node_data(path)` -> Returns DataFrame with cols: ['time', 'lat', 'lon', 'depth', 'elev', ...]
    - `calculate_wave_velocity(df)` -> Returns DataFrame
    - `get_elevation_difference(df1, df2)` -> Returns DataFrame
    - `filter_by_point(df, lat, lon)` -> Returns DataFrame
    - `visualize_map(df, value_col, title)` -> Returns Figure (Auto-Plots)
    
    CRITICAL RULES:
    1. **ASSIGN VARIABLES:** You MUST assign tool outputs to variables.
       ❌ WRONG: `get_schism_node_data(netcdf_path)` (Data is lost!)
       ✅ CORRECT: `df = get_schism_node_data(netcdf_path)`
       
    2. **CHAINING:** Pass the variable from step 1 into step 2.
       ✅ CORRECT:
       ```python
       df = get_schism_node_data(netcdf_path)
       df = calculate_wave_velocity(df) # Update the dataframe
       ```

    3. **PLOTTING:** 
       ❌ DO NOT write manual plotting code like `plt.plot()` or `df.plot()`.
       ✅ ALWAYS use the `visualize_map` tool for spatial data.
       Example: `visualize_map(df, 'elev_diff', 'Elevation Difference')`
       Note: You still need to call `plt.show()` at the end.
    
    4. **Imports:** `import pandas as pd`, `import geopandas as gpd`, `import matplotlib.pyplot as plt`
    
    5. **Output:** Output ONLY valid Python code.

    6. **FILE LOADING BAN:** 
       ❌ NEVER use `xr.open_dataset('filename.nc')` or any string path.
       ✅ ALWAYS use `netcdf_path` or `scenario_path` variables directly.
       Example: `df = get_schism_node_data(scenario_path)`

    7. **RECIPE: Maximum Wave Velocity Map**
       If the user asks for "Maximum wave velocity map":
       ```python
       # 1. Load Data
       df = get_schism_node_data(scenario_path) 
       # 2. Calculate Velocity
       df = calculate_wave_velocity(df)
       # 3. Aggregate (Max over time per node)
       # IMPORTANT: as_index=False keeps lat/lon as columns
       max_df = df.groupby(['lat', 'lon'], as_index=False)['wave_velocity'].max()
       # 4. Plot
       fig = plot_unstructured(max_df['wave_velocity'], max_df['lon'], max_df['lat'])
       plt.show()
       ```
    """
    
    # Robustly handle steps, ensuring they are strings
    steps = plan.get("steps", [])
    cleaned_steps = []
    for step in steps:
        if isinstance(step, dict):
            # Try to find a description or just dump the dict
            cleaned_steps.append(step.get("description", step.get("step", str(step))))
        else:
            cleaned_steps.append(str(step))
            
    plan_str = "\n".join(cleaned_steps)
    
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
