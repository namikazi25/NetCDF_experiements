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
    
    ### SCHISM VARIABLE KNOWLEDGE BASE:
    1. **Surface Elevation**: Use `elev` (NOT `zcor`).
    2. **Bathymetry/Depth**: Use `depth`.
    3. **Coordinates**: Use `SCHISM_hgrid_node_x` and `SCHISM_hgrid_node_y`.
    
    ### DERIVED VARIABLES (CALCULATIONS REQUIRED):
    *The file stores vector components. You MUST calculate magnitudes for these:*
    
    1. **"Wave Height" / "Significant Wave Height"**:
       - LOGIC: Magnitude of `wsh_x` and `wsh_y`.
       - CODE HINT: `np.sqrt(ds['wsh_x']**2 + ds['wsh_y']**2)`
       
    2. **"Current Speed" / "Velocity Magnitude"**:
       - LOGIC: Magnitude of `hvel_x` and `hvel_y`.
       - CODE HINT: `np.sqrt(ds['hvel_x']**2 + ds['hvel_y']**2)`
    
    3. **"Wind Speed"**:
       - LOGIC: Magnitude of `wind_x` and `wind_y` (if present).

    ### COMPARISON MODE RULES:
    *Triggered by words like "difference", "change", "impact", "delta"*
    1. Assume TWO datasets are loaded: `ds_base` (Baseline) and `ds_comp` (Scenario).
    2. Step 1: "Ensure both `ds_base` and `ds_comp` are loaded."
    3. Step 2: "Calculate difference: `diff = ds_comp['var'] - ds_base['var']`."
    4. Step 3: "Plot the `diff` variable."

    ### DEFAULT BEHAVIOR:
    If `scenario_path` is active (Comparison Mode):
      * **Standard Queries:** If user asks "What is the max depth?", you MUST calculate it for **BOTH** `ds_base` and `ds_comp` and report both numbers.
      * **Plot Queries:** If user asks "Plot velocity", generate **THREE** plots if possible: Base, Scenario, and Difference.

    ### CRITICAL RULES:
    1. **Never** say a variable is missing without checking if it can be derived from components (e.g. wsh_x, wsh_y).
    2. If asking for a map, use the helper: `plot_unstructured(variable, ds['SCHISM_hgrid_node_x'], ds['SCHISM_hgrid_node_y'])`.
    3. Do NOT ask to print `ds.info()`.
    4. DO NOT import 'schism'. It does not exist. Use xarray.
    5. DO NOT try to calculate triangulation manually. Use `plot_unstructured`.
    6. DO NOT use absolute paths.
    
    OUTPUT FORMAT:
    {
        "thought": "Reasoning...",
        "steps": ["Step 1...", "Step 2..."]
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
