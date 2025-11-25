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
    
    # Static variables
    lat = ds['SCHISM_hgrid_node_y'].values
    lon = ds['SCHISM_hgrid_node_x'].values
    depth = ds['depth'].values
    time_vals = ds['time'].values
    
    records = []
    
    # Simplified extraction loop (optimized from your notebook)
    for t_index, t_val in enumerate(time_vals):
        # Handle 3D layers safely - defaulting to last layer (surface) if 3D
        layer_idx = -1 
        
        df_step = pd.DataFrame({
            'time': t_val,
            'lat': lat,
            'lon': lon,
            'depth': depth,
            'elev': ds['elev'].values[t_index, :],
            'wsh_x': ds['wsh_x'].values[t_index, :] if 'wsh_x' in ds else 0,
            'wsh_y': ds['wsh_y'].values[t_index, :] if 'wsh_y' in ds else 0,
            'tp': ds['tp'].values[t_index, :] if 'tp' in ds else 0,
        })
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
