import xarray as xr
import pandas as pd
import numpy as np
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from domain_tools import get_schism_node_data

def debug_schism_loading():
    nc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "schouts_2.nc")
    print(f"Loading {nc_path}...")
    
    ds = xr.open_dataset(nc_path)
    print("Dataset Dimensions:")
    print(ds.dims)
    
    print("\nVariable Shapes:")
    try:
        lat = ds['SCHISM_hgrid_node_y'].values
        lon = ds['SCHISM_hgrid_node_x'].values
        depth = ds['depth'].values
        print(f"Lat: {lat.shape}")
        print(f"Lon: {lon.shape}")
        print(f"Depth: {depth.shape}")
        
        if 'elev' in ds:
            print(f"Elev: {ds['elev'].shape}")
        else:
            print("Elev not found")
            
        if 'wsh_x' in ds:
            print(f"wsh_x: {ds['wsh_x'].shape}")
            
    except Exception as e:
        print(f"Error inspecting variables: {e}")
    finally:
        ds.close()

    print("\nAttempting get_schism_node_data...")
    try:
        df = get_schism_node_data(nc_path)
        print("Success!")
        print(df.head())
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()
        
    print("\nAttempting get_dataset_summary...")
    try:
        from domain_tools import get_dataset_summary
        summary = get_dataset_summary(nc_path)
        print("Summary Success!")
        print(summary)
    except Exception as e:
        print(f"Summary FAILURE: {e}")

if __name__ == "__main__":
    debug_schism_loading()
