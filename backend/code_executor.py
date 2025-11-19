import io
import contextlib
import base64
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import scipy
import os

def plot_unstructured(variable, node_x, node_y, title="Unstructured Mesh Plot"):
    """
    Helper function to plot unstructured grid data (SCHISM).
    variable: 1D array of values at nodes
    node_x: 1D array of x-coordinates
    node_y: 1D array of y-coordinates
    """
    try:
        plt.figure(figsize=(10, 8))
        plt.tripcolor(node_x, node_y, variable, shading='flat')
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

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code_string, {}, local_env)
            
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
