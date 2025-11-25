import inspect
from domain_tools import (
    get_schism_node_data, 
    calculate_wave_velocity, 
    get_elevation_difference, 
    filter_by_time_window,
    # New imports
    filter_by_point,
    filter_by_bbox,
    get_dataset_summary,
    visualize_map # <--- New Tool
]

# List of active tools
ACTIVE_TOOLS = [
    get_schism_node_data,
    calculate_wave_velocity,
    get_elevation_difference,
    filter_by_time_window,
    filter_by_point,
    filter_by_bbox,
    get_dataset_summary,
    visualize_map
]

def get_tools_map():
    """Returns a dictionary mapping function names to the actual function objects."""
    return {func.__name__: func for func in ACTIVE_TOOLS}

def generate_tool_documentation():
    """
    Auto-generates a prompt section describing available tools.
    The LLM uses this to decide which tool to pick.
    """
    doc_str = "### 🛠️ AVAILABLE DOMAIN TOOLS:\n"
    doc_str += "You have access to the following Python functions. USE THEM instead of writing raw extraction logic.\n\n"
    
    for func in ACTIVE_TOOLS:
        name = func.__name__
        # Get the docstring (clean up indentation)
        desc = inspect.getdoc(func)
        sig = inspect.signature(func)
        
        doc_str += f"#### `{name}{sig}`\n"
        doc_str += f"{desc}\n\n"
        
    return doc_str
