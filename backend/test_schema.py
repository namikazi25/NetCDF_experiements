import os
import sys
import json

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from schema_registry import analyze_netcdf_schema

def test_schema():
    nc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "schouts_1.nc")
    print(f"Analyzing {nc_path}...")
    
    schema = analyze_netcdf_schema(nc_path)
    
    if "error" in schema:
        print(f"FAILURE: {schema['error']}")
    else:
        print("SUCCESS!")
        print(json.dumps(schema, indent=2, default=str))

if __name__ == "__main__":
    test_schema()
