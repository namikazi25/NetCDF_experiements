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
    - `scenario_path`: Path to the Scenario file (None if single mode).
    - `ds`: The dataset is ALREADY LOADED as `ds = xr.open_dataset(netcdf_path)`
    
    CRITICAL RULES:
    1. **Context Loaded:** The variables `ds` (or `ds_base`) and `ds_comp` are ALREADY loaded for you.
       ❌ WRONG: `ds = xr.open_dataset(netcdf_path)` or `ds = xr.open_dataset('file.nc')`
       ✅ CORRECT: Just use `ds` or `ds_base` directly. They are already in memory.
       - DO NOT write any `xr.open_dataset(...)` calls in your code.
       - Use `ds_base` and `ds_comp` directly for analysis.
    
    2. **Data Processing:**
       - Convert to DataFrame: `df = ds.to_dataframe().reset_index()`
       - **IMPORTANT**: Inspect actual column names using `df.columns` or `list(ds.coords)`
       - Do NOT assume column names like 'lon', 'lat', 'x', 'y' exist
       - Common SCHISM coordinate names: 'SCHISM_hgrid_node_x', 'SCHISM_hgrid_node_y'
    
    3. **Memory Management (CRITICAL):**
       - NetCDF files can be HUGE. Do NOT load entire arrays into memory.
       - For time-series data, select specific time steps: `ds.isel(time=-1)` or `ds.isel(time=0)`
       - For spatial subsets, use `.sel()` or `.isel()` to limit data
       - Example: `df = ds.isel(time=-1).to_dataframe().reset_index()`
    
    4. **Plotting with GeoPandas (STRICT):**
       - **You MUST use `geopandas` for all map plots.**
       - **CRITICAL:** Always use `markersize=5` to properly visualize mesh nodes.
       - First, inspect the DataFrame to find coordinate columns:
         ```python
         print("Available columns:", df.columns.tolist())
         # Look for columns containing 'x', 'y', 'lon', 'lat', or 'node'
         ```
       - Then create GeoDataFrame with actual column names:
         ```python
         import geopandas as gpd
         # Use the ACTUAL coordinate column names from the dataset
         gdf = gpd.GeoDataFrame(
             df, 
             geometry=gpd.points_from_xy(df['SCHISM_hgrid_node_x'], df['SCHISM_hgrid_node_y']),
             crs="EPSG:4326"
         )
         fig, ax = plt.subplots(figsize=(10, 8))
         # CRITICAL: Use markersize=5 to show individual nodes
         gdf.plot(column='variable_name', cmap='viridis', legend=True, ax=ax, markersize=5)
         plt.show()
         ```
    
    5. **Imports:** ALWAYS start your code with:
       ```python
       import xarray as xr
       import pandas as pd
       import numpy as np
       import geopandas as gpd
       import matplotlib.pyplot as plt
       ```
    
    6. **Output:** Output ONLY valid Python code. No explanations.
    """
    
    # Handle steps that might be strings, dicts, or other objects
    steps = plan.get("steps", [])
    if steps and isinstance(steps[0], dict):
        # If steps are dicts, extract a text representation
        plan_str = "\n".join([str(step) for step in steps])
    elif steps and isinstance(steps[0], str):
        # If steps are already strings, join them
        plan_str = "\n".join(steps)
    else:
        # Fallback: convert whatever we have to string
        plan_str = str(steps)
    
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
        
        # Check if the error is due to hardcoded filename
        if "No such file or directory" in error_msg and ".nc" in error_msg:
            error_msg += "\n\n⚠️ CRITICAL: You are hardcoding the filename! Use the variable `netcdf_path` instead of writing the filename as a string."
        
        if attempt < max_retries - 1:
            fix_prompt = f"""The code failed with this error:
            {error_msg}
            
            Analyze why, fix the code, and output the FULL corrected code block.
            Remember: The dataset is ALREADY LOADED as `ds`. Do NOT try to load it again.
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
