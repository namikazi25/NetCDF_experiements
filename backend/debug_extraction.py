
import os
import sys
import pandas as pd
import xarray as xr

# Add current directory to sys.path
sys.path.append(os.getcwd())

from domain_tools import get_schism_node_data

def debug_extraction():
    # Path to a file that exists (Baseline or Scenario)
    # Try schouts_2.nc (Baseline) or schouts_1.nc (Scenario)
    path = "uploads/schouts_2.nc"
    if not os.path.exists(path):
        path = "uploads/schouts_1.nc"
    
    if not os.path.exists(path):
        print("No NetCDF file found in uploads/ to test.")
        return

    print(f"Testing extraction on: {path}")
    
    try:
        df = get_schism_node_data(path)
        print("\n--- DataFrame Info ---")
        print(df.info())
        print("\n--- First 5 Rows ---")
        print(df.head())
        print("\n--- Columns ---")
        print(df.columns.tolist())
        
        # Check for critical columns
        required = ['time', 'lat', 'lon', 'elev', 'layer']
        missing = [c for c in required if c not in df.columns]
        
        if missing:
            print(f"\n❌ MISSING COLUMNS: {missing}")
        else:
            print("\n✅ All required columns present.")
            
    except Exception as e:
        print(f"\n❌ ERROR during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_extraction()
