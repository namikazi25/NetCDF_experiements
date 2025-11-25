
import pandas as pd
import numpy as np
import xarray as xr
import os
import sys

# Add current directory to sys.path to import domain_tools
sys.path.append(os.getcwd())
from domain_tools import get_schism_node_data, calculate_wave_velocity

def user_logic_simulation(netcdf_path):
    """
    Simulates the logic found in 'netcdf_query_for_llm_check (2).py'
    specifically for the 'Maximum wave velocity' query.
    """
    print("--- Running User Logic ---")
    ds = xr.open_dataset(netcdf_path)
    
    # User script logic (vectorized version from generate_full_node_table2)
    # They construct a dataframe. We will try to mimic the core calculation.
    
    # 1. Extract arrays
    wsh_x = ds['wsh_x'].values # (time, node)
    wsh_y = ds['wsh_y'].values # (time, node)
    
    # 2. Calculate Velocity
    # scenario_table['wave_velocity']=np.sqrt(scenario_table['wsh_x']**2+scenario_table['wsh_y']**2)
    wave_velocity = np.sqrt(wsh_x**2 + wsh_y**2)
    
    # 3. Max over time (The user groups by lat/lon and takes max)
    # Since wsh_x is (time, node), we can just take max over axis 0 (time)
    # to get the max velocity per node.
    max_vel_per_node = np.max(wave_velocity, axis=0)
    
    print(f"User Logic Stats:")
    print(f"  Min: {np.min(max_vel_per_node)}")
    print(f"  Max: {np.max(max_vel_per_node)}")
    print(f"  Mean: {np.mean(max_vel_per_node)}")
    
    ds.close()
    return max_vel_per_node

def system_logic_simulation(netcdf_path):
    """
    Runs the system's domain_tools logic.
    """
    print("\n--- Running System Logic ---")
    
    # 1. Load Data
    df = get_schism_node_data(netcdf_path)
    
    # 2. Calculate Velocity
    df = calculate_wave_velocity(df)
    
    # 3. Max over time (Group by lat/lon)
    # Note: get_schism_node_data rounds coordinates to 6 decimals now.
    max_vel_df = df.groupby(['lat', 'lon'])['wave_velocity'].max()
    
    print(f"System Logic Stats:")
    print(f"  Min: {max_vel_df.min()}")
    print(f"  Max: {max_vel_df.max()}")
    print(f"  Mean: {max_vel_df.mean()}")
    
    return max_vel_df.values

if __name__ == "__main__":
    # Check Scenario File (schouts_1.nc)
    path = "uploads/schouts_1.nc"
    if not os.path.exists(path):
        print(f"File not found: {path}")
    else:
        user_vals = user_logic_simulation(path)
        sys_vals = system_logic_simulation(path)
        
        # Compare
        print("\n--- Comparison ---")
        diff = np.abs(np.mean(user_vals) - np.mean(sys_vals))
        print(f"Difference in Means: {diff}")
        
        if diff < 1e-5:
            print("✅ MATCH: Data calculation is consistent.")
        else:
            print("❌ MISMATCH: Data calculation differs.")
