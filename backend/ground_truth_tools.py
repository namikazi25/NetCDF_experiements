"""
Ground Truth Tool Registry for NetCDF Queries

This module provides function tools that match proven query patterns from the reference notebook.
Each tool encapsulates a specific analysis pattern with exact implementation.
"""

import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import io
import base64


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def generate_full_node_table(netcdf_path: str) -> pd.DataFrame:
    """
    Generate a complete node-based table from SCHISM NetCDF.
    Includes static node info and dynamic variables over time.
    Returns a long-format DataFrame with time parsing.
    """
    ds = xr.open_dataset(netcdf_path)

    lat = ds['SCHISM_hgrid_node_y'].values
    lon = ds['SCHISM_hgrid_node_x'].values
    depth = ds['depth'].values
    node_bottom_index = ds['node_bottom_index'].values
    time = ds['time'].values

    records = []

    for t_index, t_val in enumerate(time):
        for layer in range(ds['hvel_x'].shape[2]):
            df_layer = pd.DataFrame({
                'time': t_val,
                'lat': lat,
                'lon': lon,
                'layer': layer,
                'depth': depth,
                'node_bottom_index': node_bottom_index,
                'zcor': ds['zcor'].values[t_index, :, layer],
                'elev': ds['elev'].values[t_index, :],
                'hvel_x': ds['hvel_x'].values[t_index, :, layer],
                'hvel_y': ds['hvel_y'].values[t_index, :, layer],
                'wsh_x': ds['wsh_x'].values[t_index, :],
                'wsh_y': ds['wsh_y'].values[t_index, :],
                'tp': ds['tp'].values[t_index, :],
                'WWM_1': ds['WWM_1'].values[t_index, :],
                'WWM_3': ds['WWM_3'].values[t_index, :],
                'WWM_9': ds['WWM_9'].values[t_index, :]
            })
            records.append(df_layer)

    full_df = pd.concat(records, ignore_index=True)
    full_df['time'] = pd.to_datetime(full_df['time'])
    full_df['year'] = full_df['time'].dt.year
    full_df['month'] = full_df['time'].dt.month
    full_df['day'] = full_df['time'].dt.day
    full_df['hour'] = full_df['time'].dt.hour
    full_df['am_pm_time'] = full_df['time'].dt.strftime("%I:%M %p")
    full_df['day_time'] = full_df['am_pm_time'].apply(lambda v: v[0:2] + '-' + v[-2:])

    ds.close()
    return full_df


def fig_to_base64() -> str:
    """Capture current matplotlib figure as base64 string."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

TOOL_REGISTRY = {}


def register_tool(name: str, description: str, parameters: dict):
    """Decorator to register a function as a callable tool."""
    def decorator(func):
        TOOL_REGISTRY[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": func
        }
        return func
    return decorator


# ----------------------------------------------------------------------------
# Tool 1: Parameter Shape Table
# ----------------------------------------------------------------------------
@register_tool(
    name="get_parameter_shapes",
    description="Show the shape and size of all parameters/variables in a NetCDF file",
    parameters={
        "netcdf_path": {"type": "string", "description": "Path to the NetCDF file"}
    }
)
def get_parameter_shapes(netcdf_path: str) -> Dict[str, Any]:
    """Ground truth: Show shape and size of all parameters."""
    from netCDF4 import Dataset
    
    data = Dataset(netcdf_path)
    var_list = []
    shape_list = []
    
    params = list(data.variables.keys())
    for param in params:
        var_list.append(param)
        shape_list.append(data.variables[param][:].shape)
    
    data.close()
    
    shape_table = pd.DataFrame({'Parameter': var_list, 'Shape': shape_list})
    
    return {
        "table": shape_table.to_dict('records'),
        "summary": f"Found {len(var_list)} parameters in the file"
    }


# ----------------------------------------------------------------------------
# Tool 2: Average Water Depth
# ----------------------------------------------------------------------------
@register_tool(
    name="calculate_average_depth",
    description="Calculate and plot average water depth for each location (lat/lon)",
    parameters={
        "netcdf_path": {"type": "string", "description": "Path to the NetCDF file"},
        "generate_plot": {"type": "boolean", "description": "Whether to generate a plot", "default": True}
    }
)
def calculate_average_depth(netcdf_path: str, generate_plot: bool = True) -> Dict[str, Any]:
    """Ground truth: Average water depth query."""
    table = generate_full_node_table(netcdf_path)
    depth_data = table.groupby(['lat', 'lon'], as_index=False)['depth'].mean()
    
    result = {
        "table": depth_data.to_dict('records'),
        "stats": {
            "mean_depth": float(depth_data['depth'].mean()),
            "max_depth": float(depth_data['depth'].max()),
            "min_depth": float(depth_data['depth'].min()),
            "num_points": len(depth_data)
        }
    }
    
    if generate_plot:
        gdf = gpd.GeoDataFrame(
            depth_data,
            geometry=gpd.points_from_xy(depth_data['lon'], depth_data['lat']),
            crs="EPSG:4326"
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf.plot(ax=ax, column='depth', cmap='viridis', markersize=5, legend=True)
        ax.set_title("Geospatial Plot of Points Colored by Depth", fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        result["image"] = fig_to_base64()
    
    return result


# ----------------------------------------------------------------------------
# Tool 3: Min/Max Depth Locations
# ----------------------------------------------------------------------------
@register_tool(
    name="find_min_max_depth",
    description="Find locations with minimum and maximum depth values",
    parameters={
        "netcdf_path": {"type": "string", "description": "Path to the NetCDF file"},
        "generate_plot": {"type": "boolean", "description": "Whether to generate a plot", "default": True}
    }
)
def find_min_max_depth(netcdf_path: str, generate_plot: bool = True) -> Dict[str, Any]:
    """Ground truth: Find min and max depth locations."""
    table = generate_full_node_table(netcdf_path)
    
    depth_max = table[table['depth'] == table['depth'].max()][['lat', 'lon', 'depth']].drop_duplicates()
    depth_min = table[table['depth'] == table['depth'].min()][['lat', 'lon', 'depth']].drop_duplicates()
    depth_main_max = pd.concat([depth_max, depth_min], ignore_index=True)
    
    result = {
        "max_depth": depth_max.to_dict('records'),
        "min_depth": depth_min.to_dict('records'),
        "combined_table": depth_main_max.to_dict('records')
    }
    
    if generate_plot:
        gdf = gpd.GeoDataFrame(
            depth_main_max,
            geometry=gpd.points_from_xy(depth_main_max['lon'], depth_main_max['lat']),
            crs="EPSG:4326"
        )
        
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = ['red' if d == gdf['depth'].max() else 'green' for d in gdf['depth']]
        gdf.plot(ax=ax, color=colors, markersize=80)
        
        for x, y, depth in zip(gdf.geometry.x, gdf.geometry.y, gdf['depth']):
            ax.text(x, y - 0.01, f"{depth:.2f} m", fontsize=8, ha='center',
                    va='top', rotation=90, weight='bold', color='black')
        
        ax.set_title("Max & Min Depth Points", fontsize=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        result["image"] = fig_to_base64()
    
    return result


# ----------------------------------------------------------------------------
# Tool 4: Wave Velocity (Maximum)
# ----------------------------------------------------------------------------
@register_tool(
    name="calculate_max_wave_velocity",
    description="Calculate maximum wave orbital velocity (from wsh_x and wsh_y components) for each location",
    parameters={
        "netcdf_path": {"type": "string", "description": "Path to the NetCDF file"},
        "generate_plot": {"type": "boolean", "description": "Whether to generate a plot", "default": True}
    }
)
def calculate_max_wave_velocity(netcdf_path: str, generate_plot: bool = True) -> Dict[str, Any]:
    """Ground truth: Maximum wave velocity map."""
    table = generate_full_node_table(netcdf_path)
    table['wave_velocity'] = np.sqrt(table['wsh_x']**2 + table['wsh_y']**2)
    wave_vel = table.groupby(['lat', 'lon'], as_index=False)['wave_velocity'].max()
    
    result = {
        "table": wave_vel.to_dict('records'),
        "stats": {
            "max_velocity": float(wave_vel['wave_velocity'].max()),
            "mean_velocity": float(wave_vel['wave_velocity'].mean()),
            "num_points": len(wave_vel)
        }
    }
    
    if generate_plot:
        gdf = gpd.GeoDataFrame(
            wave_vel,
            geometry=gpd.points_from_xy(wave_vel['lon'], wave_vel['lat']),
            crs="EPSG:4326"
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf.plot(ax=ax, column='wave_velocity', cmap='viridis', markersize=5, legend=True)
        ax.set_title("Maximum Wave Velocity", fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        result["image"] = fig_to_base64()
    
    return result


# ----------------------------------------------------------------------------
# Tool 5: Elevation Difference (Comparison)
# ----------------------------------------------------------------------------
@register_tool(
    name="calculate_elevation_difference",
    description="Calculate and plot the difference in surface elevation between two NetCDF files (base - scenario)",
    parameters={
        "base_path": {"type": "string", "description": "Path to the baseline NetCDF file"},
        "scenario_path": {"type": "string", "description": "Path to the scenario NetCDF file"},
        "generate_plot": {"type": "boolean", "description": "Whether to generate a plot", "default": True}
    }
)
def calculate_elevation_difference(base_path: str, scenario_path: str, generate_plot: bool = True) -> Dict[str, Any]:
    """Ground truth: Elevation difference between two files."""
    base_table = generate_full_node_table(base_path)
    scenario_table = generate_full_node_table(scenario_path)
    
    base_ele = base_table.groupby(['lat', 'lon'], as_index=False)['elev'].mean()
    scenario_ele = scenario_table.groupby(['lat', 'lon'], as_index=False)['elev'].mean()
    
    merge_elev_table = base_ele.merge(
        scenario_ele,
        on=['lat', 'lon'],
        how='inner',
        suffixes=('_base', '_scenario')
    )
    
    merge_elev_table['elev_diff'] = merge_elev_table['elev_base'] - merge_elev_table['elev_scenario']
    final_table = merge_elev_table[['lat', 'lon', 'elev_diff']]
    
    result = {
        "table": final_table.to_dict('records'),
        "stats": {
            "mean_diff": float(final_table['elev_diff'].mean()),
            "max_diff": float(final_table['elev_diff'].max()),
            "min_diff": float(final_table['elev_diff'].min()),
            "std_diff": float(final_table['elev_diff'].std())
        }
    }
    
    if generate_plot:
        gdf = gpd.GeoDataFrame(
            final_table,
            geometry=gpd.points_from_xy(final_table['lon'], final_table['lat']),
            crs="EPSG:4326"
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf.plot(ax=ax, column='elev_diff', cmap='RdBu_r', markersize=5, legend=True)
        ax.set_title("Elevation Difference (Base - Scenario)", fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        result["image"] = fig_to_base64()
    
    return result


# ----------------------------------------------------------------------------
# Tool 6: Point Time Series
# ----------------------------------------------------------------------------
@register_tool(
    name="get_point_time_series",
    description="Show change of a variable over time for a specific point (lat/lon)",
    parameters={
        "netcdf_path": {"type": "string", "description": "Path to the NetCDF file"},
        "lat": {"type": "number", "description": "Latitude of the point"},
        "lon": {"type": "number", "description": "Longitude of the point"},
        "variable": {"type": "string", "description": "Variable to analyze (e.g., 'elev')", "default": "elev"},
        "tolerance": {"type": "number", "description": "Tolerance for lat/lon matching", "default": 1e-6},
        "generate_plot": {"type": "boolean", "description": "Whether to generate a plot", "default": True}
    }
)
def get_point_time_series(
    netcdf_path: str,
    lat: float,
    lon: float,
    variable: str = "elev",
    tolerance: float = 1e-6,
    generate_plot: bool = True
) -> Dict[str, Any]:
    """Ground truth: Time series for a specific point."""
    table = generate_full_node_table(netcdf_path)
    
    filtered = table[
        (table['lat'].between(lat - tolerance, lat + tolerance)) &
        (table['lon'].between(lon - tolerance, lon + tolerance))
    ][['time', variable, 'hour', 'day_time']].drop_duplicates()
    
    result = {
        "table": filtered.to_dict('records'),
        "point": {"lat": lat, "lon": lon},
        "variable": variable,
        "num_timesteps": len(filtered)
    }
    
    if generate_plot and len(filtered) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        filtered.plot(x='day_time', y=variable, ax=ax, legend=True)
        ax.set_title(f"{variable.title()} over Time at ({lat:.4f}, {lon:.4f})", fontsize=14)
        ax.set_xlabel("Time")
        ax.set_ylabel(variable.title())
        
        result["image"] = fig_to_base64()
    
    return result


# ----------------------------------------------------------------------------
# Tool 7: Maximum Variable Value
# ----------------------------------------------------------------------------
@register_tool(
    name="find_max_variable_location",
    description="Find the location(s) where a variable has its maximum value",
    parameters={
        "netcdf_path": {"type": "string", "description": "Path to the NetCDF file"},
        "variable": {"type": "string", "description": "Variable to analyze (e.g., 'tp', 'elev')"}
    }
)
def find_max_variable_location(netcdf_path: str, variable: str) -> Dict[str, Any]:
    """Ground truth: Find max value location for any variable."""
    table = generate_full_node_table(netcdf_path)
    
    max_rows = table[table[variable] == table[variable].max()][['lat', 'lon', 'day_time', variable]].drop_duplicates()
    
    return {
        "table": max_rows.to_dict('records'),
        "max_value": float(table[variable].max()),
        "variable": variable
    }


# ----------------------------------------------------------------------------
# Tool 8: Velocity Exceeds Average
# ----------------------------------------------------------------------------
@register_tool(
    name="find_velocity_above_average",
    description="Find locations where wave orbital velocity exceeds the average velocity",
    parameters={
        "netcdf_path": {"type": "string", "description": "Path to the NetCDF file"},
        "top_n": {"type": "integer", "description": "Return only top N locations (0 for all)", "default": 0},
        "generate_plot": {"type": "boolean", "description": "Whether to generate a plot", "default": True}
    }
)
def find_velocity_above_average(
    netcdf_path: str,
    top_n: int = 0,
    generate_plot: bool = True
) -> Dict[str, Any]:
    """Ground truth: Locations where velocity exceeds average."""
    table = generate_full_node_table(netcdf_path)
    table['wave_velocity'] = np.sqrt(table['wsh_x']**2 + table['wsh_y']**2)
    wave_vel = table.groupby(['lat', 'lon'], as_index=False)['wave_velocity'].mean()
    
    avg_velocity = wave_vel['wave_velocity'].mean()
    wave_vel_filter = wave_vel[wave_vel['wave_velocity'] > avg_velocity].sort_values(
        by='wave_velocity', ascending=False
    )
    
    if top_n > 0:
        wave_vel_filter['rank'] = wave_vel_filter['wave_velocity'].rank(method='dense', ascending=False)
        wave_vel_filter = wave_vel_filter[wave_vel_filter['rank'] <= top_n]
        output_df = wave_vel_filter[['lat', 'lon', 'wave_velocity']]
        title_suffix = f" (Top {top_n})"
    else:
        output_df = wave_vel_filter
        title_suffix = ""
    
    result = {
        "table": output_df.to_dict('records'),
        "average_velocity": float(avg_velocity),
        "num_points_above_avg": len(output_df)
    }
    
    if generate_plot and len(output_df) > 0:
        gdf = gpd.GeoDataFrame(
            output_df,
            geometry=gpd.points_from_xy(output_df['lon'], output_df['lat']),
            crs="EPSG:4326"
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf.plot(ax=ax, column='wave_velocity', cmap='viridis', markersize=5, legend=True)
        ax.set_title(f"Wave Velocity Above Average{title_suffix}", fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        result["image"] = fig_to_base64()
    
    return result


# ----------------------------------------------------------------------------
# Tool 9: Time-Filtered Elevation
# ----------------------------------------------------------------------------
@register_tool(
    name="calculate_time_filtered_elevation",
    description="Calculate average water elevation between specific hours",
    parameters={
        "netcdf_path": {"type": "string", "description": "Path to the NetCDF file"},
        "start_hour": {"type": "integer", "description": "Start hour (0-23)", "default": 10},
        "end_hour": {"type": "integer", "description": "End hour (0-23)", "default": 15},
        "generate_plot": {"type": "boolean", "description": "Whether to generate a plot", "default": True},
        "include_summary_table": {"type": "boolean", "description": "Include pivot table summary", "default": True},
        "include_time_analysis": {"type": "boolean", "description": "Include time-wise analysis", "default": True}
    }
)
def calculate_time_filtered_elevation(
    netcdf_path: str,
    start_hour: int = 10,
    end_hour: int = 15,
    generate_plot: bool = True,
    include_summary_table: bool = True,
    include_time_analysis: bool = True
) -> Dict[str, Any]:
    """Ground truth: Time-filtered elevation analysis."""
    table = generate_full_node_table(netcdf_path)
    
    filter_data = table[(table['hour'] >= start_hour) & (table['hour'] <= end_hour)]
    filter_data_elev = filter_data.groupby(['lat', 'lon'], as_index=False)['elev'].mean()
    
    result = {
        "table": filter_data_elev.to_dict('records'),
        "time_range": {"start_hour": start_hour, "end_hour": end_hour},
        "num_points": len(filter_data_elev)
    }
    
    if include_summary_table:
        elev_summary = filter_data.pivot_table(
            index=['lat', 'lon'],
            columns=['day_time'],
            values='elev',
            aggfunc='mean'
        )
        result["summary_pivot"] = elev_summary.reset_index().to_dict('records')
    
    if include_time_analysis:
        time_wise = filter_data.groupby('day_time')['elev'].mean()
        result["time_wise_analysis"] = time_wise.to_dict()
    
    if generate_plot:
        gdf = gpd.GeoDataFrame(
            filter_data_elev,
            geometry=gpd.points_from_xy(filter_data_elev['lon'], filter_data_elev['lat']),
            crs="EPSG:4326"
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        gdf.plot(ax=ax, column='elev', cmap='viridis', markersize=5, legend=True)
        ax.set_title(f"Average Elevation ({start_hour}:00 - {end_hour}:00)", fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        result["image"] = fig_to_base64()
    
    return result


# ----------------------------------------------------------------------------
# Tool 10: Bounding Box Filter
# ----------------------------------------------------------------------------
@register_tool(
    name="analyze_bounding_box",
    description="Analyze data within a geographic bounding box, optionally filtered by time",
    parameters={
        "netcdf_path": {"type": "string", "description": "Path to the NetCDF file"},
        "lat_min": {"type": "number", "description": "Minimum latitude"},
        "lat_max": {"type": "number", "description": "Maximum latitude"},
        "lon_min": {"type": "number", "description": "Minimum longitude"},
        "lon_max": {"type": "number", "description": "Maximum longitude"},
        "variable": {"type": "string", "description": "Variable to analyze", "default": "elev"},
        "start_hour": {"type": "integer", "description": "Start hour filter (optional)", "default": None},
        "end_hour": {"type": "integer", "description": "End hour filter (optional)", "default": None},
        "generate_plot": {"type": "boolean", "description": "Whether to generate a plot", "default": True}
    }
)
def analyze_bounding_box(
    netcdf_path: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    variable: str = "elev",
    start_hour: Optional[int] = None,
    end_hour: Optional[int] = None,
    generate_plot: bool = True
) -> Dict[str, Any]:
    """Ground truth: Bounding box analysis."""
    table = generate_full_node_table(netcdf_path)
    
    # Apply time filter if specified
    if start_hour is not None and end_hour is not None:
        table = table[(table['hour'] >= start_hour) & (table['hour'] <= end_hour)]
    
    # Apply bounding box filter
    bbox_df = table[
        (table['lat'] >= lat_min) & (table['lat'] <= lat_max) &
        (table['lon'] >= lon_min) & (table['lon'] <= lon_max)
    ]
    
    bbox_df_agg = bbox_df.groupby(['lat', 'lon'], as_index=False)[variable].mean()
    
    result = {
        "table": bbox_df_agg.to_dict('records'),
        "bounding_box": {
            "lat_min": lat_min, "lat_max": lat_max,
            "lon_min": lon_min, "lon_max": lon_max
        },
        "variable": variable,
        "num_points": len(bbox_df_agg),
        "stats": {
            "mean": float(bbox_df_agg[variable].mean()),
            "max": float(bbox_df_agg[variable].max()),
            "min": float(bbox_df_agg[variable].min())
        }
    }
    
    if generate_plot and len(bbox_df_agg) > 0:
        gdf = gpd.GeoDataFrame(
            bbox_df_agg,
            geometry=gpd.points_from_xy(bbox_df_agg['lon'], bbox_df_agg['lat']),
            crs="EPSG:4326"
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        gdf.plot(ax=ax, column=variable, cmap='viridis', markersize=5, legend=True)
        ax.set_title(f"{variable.title()} in Bounding Box", fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        result["image"] = fig_to_base64()
    
    return result


# ----------------------------------------------------------------------------
# Tool 11: Average Elevation Map
# ----------------------------------------------------------------------------
@register_tool(
    name="plot_average_elevation",
    description="Plot the average surface elevation map for a NetCDF file",
    parameters={
        "netcdf_path": {"type": "string", "description": "Path to the NetCDF file"}
    }
)
def plot_average_elevation(netcdf_path: str) -> Dict[str, Any]:
    """Ground truth: Average elevation map."""
    table = generate_full_node_table(netcdf_path)
    base_ele = table.groupby(['lat', 'lon'], as_index=False)['elev'].mean()
    
    gdf = gpd.GeoDataFrame(
        base_ele,
        geometry=gpd.points_from_xy(base_ele['lon'], base_ele['lat']),
        crs="EPSG:4326"
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(ax=ax, column='elev', cmap='viridis', markersize=5, legend=True)
    ax.set_title("Average Surface Elevation", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    return {
        "table": base_ele.to_dict('records'),
        "stats": {
            "mean_elevation": float(base_ele['elev'].mean()),
            "max_elevation": float(base_ele['elev'].max()),
            "min_elevation": float(base_ele['elev'].min())
        },
        "image": fig_to_base64()
    }


# ============================================================================
# TOOL EXECUTOR
# ============================================================================

def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """Execute a registered tool by name with given parameters."""
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Tool '{tool_name}' not found. Available tools: {list(TOOL_REGISTRY.keys())}"}
    
    tool = TOOL_REGISTRY[tool_name]
    try:
        return tool["function"](**kwargs)
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}


def get_tool_definitions() -> List[Dict]:
    """Return OpenAI-compatible function definitions for all tools."""
    definitions = []
    for name, tool in TOOL_REGISTRY.items():
        definitions.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {
                    "type": "object",
                    "properties": tool["parameters"],
                    "required": [k for k, v in tool["parameters"].items() if "default" not in v]
                }
            }
        })
    return definitions


def list_tools() -> None:
    """Print all available tools."""
    print("=" * 60)
    print("AVAILABLE GROUND TRUTH TOOLS")
    print("=" * 60)
    for name, tool in TOOL_REGISTRY.items():
        print(f"\n📌 {name}")
        print(f"   {tool['description']}")
        print(f"   Parameters: {list(tool['parameters'].keys())}")


# ============================================================================
# QUERY MATCHING (Simple keyword-based for now)
# ============================================================================

QUERY_PATTERNS = {
    "get_parameter_shapes": [
        "shape", "size", "parameters", "variables", "dimensions"
    ],
    "calculate_average_depth": [
        "average depth", "mean depth", "water depth", "depth map"
    ],
    "find_min_max_depth": [
        "min depth", "max depth", "minimum depth", "maximum depth", "deepest", "shallowest"
    ],
    "calculate_max_wave_velocity": [
        "wave velocity", "wsh_x", "wsh_y", "wave orbital", "maximum velocity"
    ],
    "calculate_elevation_difference": [
        "elevation difference", "elev diff", "compare elevation", "surface elevation difference",
        "difference in surface elevation", "difference in elevation", "between two"
    ],
    "get_point_time_series": [
        "time series", "over time", "change over", "specific point", "for this point"
    ],
    "find_max_variable_location": [
        "maximum tp", "max tp", "maximum value", "where is max"
    ],
    "find_velocity_above_average": [
        "exceeds average", "above average", "top 100", "top locations"
    ],
    "calculate_time_filtered_elevation": [
        "between 10", "between hours", "10:00 AM", "3:00 PM", "time filtered"
    ],
    "analyze_bounding_box": [
        "bounding box", "within", "lat range", "lon range", "40.7 to 40.8"
    ],
    "plot_average_elevation": [
        "plot elevation", "elevation map", "surface elevation map", "average elevation"
    ]
}


def match_query_to_tool(query: str) -> Optional[str]:
    """Simple keyword matching to suggest a tool for a query."""
    query_lower = query.lower()
    
    scores = {}
    for tool_name, keywords in QUERY_PATTERNS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[tool_name] = score
    
    if scores:
        return max(scores, key=scores.get)
    return None


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    list_tools()
    
    # Test query matching
    test_queries = [
        "Show the shape and size of all parameters",
        "Calculate average water depth",
        "Find min and max depth locations",
        "Plot maximum wave velocity",
        "What is the elevation difference between base and scenario?",
        "Show elevation over time for point (40.699429, -8.756169)",
        "Find locations where velocity exceeds average",
        "Calculate elevation between 10 AM and 3 PM"
    ]
    
    print("\n" + "=" * 60)
    print("QUERY MATCHING TEST")
    print("=" * 60)
    for q in test_queries:
        matched = match_query_to_tool(q)
        print(f"\nQuery: {q}")
        print(f"Matched Tool: {matched}")
