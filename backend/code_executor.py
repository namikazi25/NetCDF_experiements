import io
import contextlib
import base64
import matplotlib.pyplot as plt
import matplotlib.tri as tri  # <--- Add this import
import xarray as xr
import numpy as np
import scipy
import os

def plot_unstructured(variable, x, y, title="Unstructured Mesh Plot", cmap=None):
    """
    Robust plotting for SCHISM/Unstructured grids.
    Automatically detects if a Diverging Colormap (Red-Blue) is needed for difference plots.
    """
    # 1. Sanitize Inputs: Convert xarray/DataArray to numpy
    if hasattr(variable, 'values'): variable = variable.values
    if hasattr(x, 'values'): x = x.values
    if hasattr(y, 'values'): y = y.values

    # 2. Handle Time Dimension: If (Time, Node), take the last time step
    if variable.ndim > 1:
        print(f"Note: Variable has dimensions {variable.shape}. Plotting final time step.")
        variable = variable[-1]

    # 3. Smart Colormap Logic
    vmin, vmax = None, None
    
    if cmap is None:
        # Check if data spans positive and negative values (indicating a Difference plot)
        # We use a threshold (1e-5) to ignore floating point noise around zero
        data_min = np.nanmin(variable)
        data_max = np.nanmax(variable)
        
        if data_min < -1e-5 and data_max > 1e-5:
             # DIVERGING DATA (e.g., Difference) -> Use Red-White-Blue
             cmap = 'RdBu_r' 
             # Force zero to be white by centering vmin/vmax
             limit = max(abs(data_min), abs(data_max))
             vmin, vmax = -limit, limit
        else:
             # SEQUENTIAL DATA (e.g., Depth, Elevation) -> Use Viridis
             cmap = 'viridis'

    try:
        plt.figure(figsize=(10, 8))
        
        # Plot with automatic or calculated vmin/vmax
        plt.tripcolor(x, y, variable, shading='flat', cmap=cmap, vmin=vmin, vmax=vmax)
        
        plt.colorbar(label="Value")
        plt.title(title)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.axis('equal')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        print(f"Error in plot_unstructured: {e}")
        return None

def execute_python_code(code_string: str, netcdf_path: str, scenario_path: str = None) -> dict:
    """
    Executes Python code in a controlled environment with access to the NetCDF file(s).
    Returns a dict with 'stdout', 'stderr', and 'images' (list of base64 strings).
    """
    # Capture stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    images = []
    
    # Pre-defined environment
    local_env = {
        "xr": xr,
        "np": np,
        "plt": plt,
        "scipy": scipy,
        "tri": tri,  # <--- Give LLM access to triangulation
        "netcdf_path": netcdf_path,
        "scenario_path": scenario_path,
        "plot_unstructured": plot_unstructured,
        "print": lambda *args, **kwargs: print(*args, file=stdout_capture, **kwargs)
    }
    
    # Store original functions to restore later
    original_savefig = plt.savefig
    original_show = plt.show
    
    # Custom savefig to capture images
    def custom_savefig(*args, **kwargs):
        buf = io.BytesIO()
        # Use the figure's savefig method directly to avoid recursion loop
        # or call the original_savefig we saved
        try:
            plt.gcf().savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            images.append(img_str)
        except Exception as e:
            print(f"Error saving plot: {e}", file=stdout_capture)
        
        # Close the figure to free memory
        plt.close()
        
    local_env["plt"].savefig = custom_savefig
    local_env["plt"].show = custom_savefig

    # Inject the loading logic automatically so the LLM can just assume ds exists
    header_code = f"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# AUTO-GENERATED LOADING
ds = xr.open_dataset('{netcdf_path}')
ds_base = ds
ds_comp = None
"""
    if scenario_path:
        header_code += f"""
ds_comp = xr.open_dataset('{scenario_path}')
print("System: Comparison Datasets Loaded.")
"""

    # Combine header + LLM code
    full_code = header_code + "\n" + code_string

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(full_code, {}, local_env)
            
        return {
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "images": images,
            "success": True
        }
    except Exception as e:
        return {
            "stdout": stdout_capture.getvalue(),
            "stderr": str(e),
            "images": images,
            "success": False
        }
    finally:
        # Restore original functions
        plt.savefig = original_savefig
        plt.show = original_show
