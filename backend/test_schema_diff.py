import os
import sys
import json
from schema_registry import analyze_netcdf_schema, format_context_for_planner

# Add current directory to path
sys.path.append(os.getcwd())

def test_schema_diff():
    file_path = "water_velocity_raster.nc"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    print(f"Testing schema diff on {file_path}...")
    schema = analyze_netcdf_schema(file_path)
    
    # Test Single Mode
    print("\n--- Single Mode Context ---")
    bundle_single = {'baseline': schema}
    context_single = format_context_for_planner(bundle_single)
    print(context_single)
    
    # Test Comparison Mode (using same file for both to simulate)
    print("\n--- Comparison Mode Context ---")
    bundle_comparison = {'baseline': schema, 'scenario': schema}
    context_comparison = format_context_for_planner(bundle_comparison)
    print(context_comparison)

if __name__ == "__main__":
    test_schema_diff()
