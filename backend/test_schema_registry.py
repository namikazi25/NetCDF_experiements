import os
import sys
import json
from schema_registry import analyze_netcdf_schema, format_schema_for_llm

# Add current directory to path
sys.path.append(os.getcwd())

def test_schema_registry():
    file_path = "water_velocity_raster.nc"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    print(f"Testing schema extraction on {file_path}...")
    schema = analyze_netcdf_schema(file_path)
    
    print("\nGenerated Schema Keys:", schema.keys())
    if "error" in schema:
        print("Error:", schema["error"])
        return

    print("\nVariables found:", list(schema["variables"].keys()))
    print("\nDerived Concepts:", schema["derived_concepts"])
    
    print("\nFormatted for LLM:")
    print("-" * 40)
    print(format_schema_for_llm(schema))
    print("-" * 40)

if __name__ == "__main__":
    test_schema_registry()
