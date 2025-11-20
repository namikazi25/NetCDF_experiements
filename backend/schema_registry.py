import xarray as xr
import numpy as np
import json

def analyze_netcdf_schema(file_path: str) -> dict:
    """
    Opens a NetCDF file and creates an LLM-friendly schema.
    It auto-detects potential vector pairs (x/y) to suggest derived variables.
    """
    try:
        ds = xr.open_dataset(file_path)
        
        schema = {
            "filename": file_path.split("/")[-1],
            "time_horizon": _get_time_info(ds),
            "variables": {},
            "derived_concepts": [] # This is the "smart" part
        }

        # 1. Catalog all raw variables
        var_names = list(ds.data_vars.keys())
        for var_name in var_names:
            da = ds[var_name]
            schema["variables"][var_name] = {
                "desc": da.attrs.get("long_name", "No description"),
                "units": da.attrs.get("units", "N/A"),
                "dims": list(da.dims),
                "shape": list(da.shape)
            }

        # 2. Auto-Detect Vector Pairs (The Logic from your user script)
        # We look for pairs like 'hvel_x'/'hvel_y' or 'wsh_x'/'wsh_y'
        processed_vectors = set()
        
        for var in var_names:
            if var.endswith("_x") or var.endswith("-x"):
                base = var[:-2] # e.g. "wsh"
                y_variant = f"{base}_y"
                
                if y_variant in var_names and base not in processed_vectors:
                    # We found a pair! Register it as a concept.
                    schema["derived_concepts"].append({
                        "concept_name": f"{base}_magnitude",
                        "components": [var, y_variant],
                        "formula": f"np.sqrt(ds['{var}']**2 + ds['{y_variant}']**2)",
                        "description": f"Calculated magnitude of {base} vector"
                    })
                    processed_vectors.add(base)

        ds.close()
        return schema

    except Exception as e:
        return {"error": f"Schema extraction failed: {str(e)}"}

def _get_time_info(ds):
    if "time" in ds:
        try:
            ts = ds["time"]
            return {
                "start": str(ts.values[0]),
                "end": str(ts.values[-1]),
                "steps": len(ts)
            }
        except:
            return "Time dimension present but unparseable"
    return "No time dimension"

def format_context_for_planner(schemas: dict) -> str:
    """
    Smartly formats context for 1 or 2 files.
    'schemas' is a dict: {'baseline': schema_dict, 'scenario': schema_dict (optional)}
    """
    
    base_schema = schemas.get('baseline')
    scen_schema = schemas.get('scenario')
    
    # Safety check for errors
    if "error" in base_schema:
        return f"Error reading baseline file: {base_schema['error']}"
    
    # --- HEADER GENERATION ---
    if scen_schema:
        # COMPARISON MODE
        context = "### DATASET CONTEXT (COMPARISON MODE):\n"
        context += f"1. **Baseline File:** `{base_schema['filename']}` (Loaded as `ds_base`)\n"
        context += f"2. **Scenario File:** `{scen_schema['filename']}` (Loaded as `ds_scen`)\n"
        
        # Check compatibility (simple check)
        if base_schema['variables'].keys() == scen_schema['variables'].keys():
             context += "\n**Structure Match:** Both files contain the same variables and dimensions.\n"
        else:
             context += "\n**Warning:** File structures differ slightly. Rely on Baseline structure.\n"
             
    else:
        # SINGLE MODE
        context = f"### DATASET CONTEXT: {base_schema['filename']}\n"
        context += "(Loaded as `ds`)\n"

    # --- SHARED SCHEMA DETAILS ---
    # We typically only need to list variables once if they are the same model output
    context += f"Time Horizon: {base_schema['time_horizon']}\n\n"
    
    # 1. Derived Concepts (Vectors)
    if base_schema["derived_concepts"]:
        context += "### CALCULABLE CONCEPTS (Vectors):\n"
        for con in base_schema["derived_concepts"]:
            context += f"- **{con['concept_name']}**: Formula: `{con['formula']}`\n"
        context += "\n"

    # 2. Raw Variables
    context += "### RAW VARIABLES:\n"
    for name, meta in base_schema["variables"].items():
        context += f"- `{name}` ({meta['dims']}): {meta['desc']}\n"

    return context

