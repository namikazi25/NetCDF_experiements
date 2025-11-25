import xarray as xr
import pandas as pd
import numpy as np

def get_schism_node_data(netcdf_path: str) -> pd.DataFrame:
    """
    EXTRACTS specific node data (elev, hvel, wsh, etc.) from a SCHISM NetCDF file.
    Use this for ANY query regarding raw data extraction or time-series creation.
    
    Returns a DataFrame with columns: 
    ['time', 'lat', 'lon', 'depth', 'elev', 'hvel_x', 'hvel_y', 'wsh_x', 'wsh_y', 'tp']
    """
    ds = xr.open_dataset(netcdf_path)
    
    # Dynamic Coordinate Detection
    lat_var = next((v for v in ['SCHISM_hgrid_node_y', 'y', 'lat', 'latitude'] if v in ds), None)
    lon_var = next((v for v in ['SCHISM_hgrid_node_x', 'x', 'lon', 'longitude'] if v in ds), None)
    
    if not lat_var or not lon_var:
        raise ValueError("Could not detect Latitude/Longitude variables.")
        
    # Use raw coordinates to match Ground Truth (no rounding)
    lat = ds[lat_var].values
    lon = ds[lon_var].values
    
    # Target size (number of nodes)
    n_nodes = len(lat)
    
    # Handle Depth (optional but common)
    depth = np.zeros(n_nodes)
    if 'depth' in ds:
        d_vals = ds['depth'].values
        if len(d_vals) == n_nodes:
            depth = d_vals
            
    time_vals = ds['time'].values
    
    records = []
    
    # 3D Layer Detection
    # We check a known 3D variable to determine the number of layers
    # User script uses 'hvel_x' to determine layers
    n_layers = 1
    if 'hvel_x' in ds and ds['hvel_x'].ndim == 3:
        n_layers = ds['hvel_x'].shape[2] # Assuming (time, node, layer)
    
    # Simplified extraction loop (optimized from your notebook)
    for t_index, t_val in enumerate(time_vals):
        
        # Loop through layers (1 to N)
        # If n_layers=1 (2D only), this runs once.
        for layer_idx in range(n_layers):
            
            # Base dictionary
            row_data = {
                'time': t_val,
                'lat': lat,
                'lon': lon,
                'depth': depth,
                'layer': np.full(n_nodes, layer_idx) # Add layer index
            }
            
            # Add variables
            target_vars = ['elev', 'wsh_x', 'wsh_y', 'tp', 'hvel_x', 'hvel_y', 'zcor']
            
            for var_name in target_vars:
                if var_name in ds:
                    # Extract time slice
                    val = ds[var_name].values[t_index, :]
                    
                    # Case 1: 2D Variable (Node,) -> Repeat for every layer
                    if val.ndim == 1 and len(val) == n_nodes:
                        row_data[var_name] = val
                        
                    # Case 2: 3D Variable (Node, Layer) -> Extract specific layer
                    elif val.ndim == 2 and val.shape[0] == n_nodes:
                        # Safety check for layer index
                        if layer_idx < val.shape[1]:
                            row_data[var_name] = val[:, layer_idx]
                        else:
                            row_data[var_name] = np.zeros(n_nodes)
                        
                    else:
                        # Fill with 0 if shape mismatch
                        row_data[var_name] = np.zeros(n_nodes)
                else:
                    row_data[var_name] = np.zeros(n_nodes)
            
            df_step = pd.DataFrame(row_data)
            records.append(df_step)

    full_df = pd.concat(records, ignore_index=True)
    
    # Feature Engineering from your notebook
    full_df['time'] = pd.to_datetime(full_df['time'])
    full_df['hour'] = full_df['time'].dt.hour
    
    ds.close()
    return full_df

def calculate_wave_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    CALCULATES the wave velocity magnitude from wsh_x and wsh_y columns.
    Use this when the user asks for "Wave Velocity" or "Orbital Velocity".
    
    Returns the DataFrame with a new column: 'wave_velocity'
    """
    if 'wsh_x' not in df.columns or 'wsh_y' not in df.columns:
        raise ValueError("DataFrame requires wsh_x and wsh_y columns.")
        
    df['wave_velocity'] = np.sqrt(df['wsh_x']**2 + df['wsh_y']**2)
    return df

def get_elevation_difference(df_base: pd.DataFrame, df_scenario: pd.DataFrame) -> pd.DataFrame:
    """
    COMPARES elevation between two datasets. 
    Use this when user asks for "Difference", "Impact", or "Change" in elevation.
    
    Returns a DataFrame containing: ['lat', 'lon', 'elev_diff']
    """
    # Group by location to ensure alignment
    # Round coordinates to ensure matching
    df_base['lat'] = df_base['lat'].round(6)
    df_base['lon'] = df_base['lon'].round(6)
    df_scenario['lat'] = df_scenario['lat'].round(6)
    df_scenario['lon'] = df_scenario['lon'].round(6)

    base_agg = df_base.groupby(['lat', 'lon'], as_index=False)['elev'].mean()
    scen_agg = df_scenario.groupby(['lat', 'lon'], as_index=False)['elev'].mean()
    
    merged = base_agg.merge(
        scen_agg, 
        on=['lat', 'lon'], 
        how='inner', 
        suffixes=('_base', '_scen')
    )
    
    merged['elev_diff'] = merged['elev_base'] - merged['elev_scen']
    return merged[['lat', 'lon', 'elev_diff']]

def filter_by_time_window(df: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
    """
    FILTERS data between specific hours (e.g., 10 AM to 3 PM).
    Use this when user specifies a time range like "during the day" or "10:00 to 15:00".
    """
    return df[(df['hour'] >= start_hour) & (df['hour'] <= end_hour)]

def filter_by_point(df: pd.DataFrame, target_lat: float, target_lon: float, tolerance: float = 1e-4) -> pd.DataFrame:
    """
    FILTERS data for a specific geographic point.
    Use this when user asks for data "at point X, Y" or "for the location...".
    Includes a small tolerance to handle floating point coordinate mismatches.
    """
    # Logic derived from Query 6 in your file
    condition = (
        (df['lat'].between(target_lat - tolerance, target_lat + tolerance)) &
        (df['lon'].between(target_lon - tolerance, target_lon + tolerance))
    )
    return df[condition]

def filter_by_bbox(df: pd.DataFrame, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> pd.DataFrame:
    """
    FILTERS data within a rectangular bounding box.
    Use this when user asks for a region defined by latitude and longitude ranges.
    """
    # Logic derived from Query 11 in your file
    condition = (
        (df['lat'] >= lat_min) & (df['lat'] <= lat_max) &
        (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
    )
    return df[condition]

def get_dataset_summary(netcdf_path: str) -> pd.DataFrame:
    """
    Returns a table of all variables in the NetCDF file, their dimensions, and shapes.
    Use this when the user asks "What is inside this file?" or "Show parameters".
    """
    ds = xr.open_dataset(netcdf_path)
    
    # Logic derived from Query 1 in your file
    data = []
    for var_name in ds.variables:
        dims = ds[var_name].dims
        shape = ds[var_name].shape
        data.append({"Variable": var_name, "Dimensions": dims, "Shape": shape})
        
    ds.close()
    return pd.DataFrame(data)

import geopandas as gpd
import matplotlib.pyplot as plt

def visualize_map(df: pd.DataFrame, value_col: str, title: str = None):
    """
    Generates a geospatial map plot for any dataframe containing 'lat' and 'lon'.
    
    Args:
        df: DataFrame with 'lat', 'lon', and the data column.
        value_col: The name of the column to color the map by (e.g., 'elev', 'elev_diff').
        title: Optional title for the plot.
    """
    # 1. Convert to GeoDataFrame (The step the LLM was missing)
    # Ensure lat/lon are numeric
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df['lon'], df['lat']), 
        crs="EPSG:4326"
    )
    
    # 2. Create the Plot (Style copied from your Ground Truth)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine colormap
    cmap = 'viridis'
    if 'diff' in value_col.lower() or 'change' in value_col.lower():
        cmap = 'RdBu_r'
        
    gdf.plot(
        ax=ax,
        column=value_col,
        cmap=cmap, 
        legend=True,
        markersize=5
    )
    
    if title:
        ax.set_title(title, fontsize=14)
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    # 3. Return figure (Executor handles plt.show())
    return fig
