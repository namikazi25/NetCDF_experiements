import xarray as xr
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import io
import base64

def generate_profile(netcdf_path):
    """
    Generates a deterministic profile of the NetCDF file.
    Returns a dictionary with metadata, stats, and a base64 encoded preview image.
    """
    try:
        ds = xr.open_dataset(netcdf_path)
    except Exception as e:
        return {"error": f"Could not open file: {e}"}
    
    # 1. Basic Metadata
    summary = {
        "filename": os.path.basename(netcdf_path),
        "variables": list(ds.data_vars.keys()),
        "dims": {k: v for k, v in ds.sizes.items()},
        "attrs": ds.attrs
    }

    # Time Horizon
    if 'time' in ds:
        try:
            summary["time_start"] = str(ds['time'][0].values)
            summary["time_end"] = str(ds['time'][-1].values)
            summary["time_steps"] = len(ds['time'])
        except:
            summary["time_error"] = "Could not parse time dimension"

    # 2. "Interesting" Stats (Hardcoded for SCHISM/Ocean models)
    # Check for common variables and compute stats
    if 'elev' in ds:
        try:
            summary['elevation_range'] = [float(ds['elev'].min()), float(ds['elev'].max())]
        except:
            pass
    
    if 'depth' in ds:
        try:
            max_depth = float(ds['depth'].max())
            summary['max_depth'] = max_depth
            
            # 3. Generate the Domain Map (Bathymetry)
            # Use robust plotting function here
            # Assume standard SCHISM node vars
            x_var = next((v for v in ['SCHISM_hgrid_node_x', 'x', 'lon'] if v in ds), None)
            y_var = next((v for v in ['SCHISM_hgrid_node_y', 'y', 'lat'] if v in ds), None)
            
            if x_var and y_var:
                x = ds[x_var]
                y = ds[y_var]
                depth = ds['depth']
                
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                # Use tricontourf for unstructured, or contourf for structured if needed
                # For SCHISM, it's unstructured, but xarray might not handle it directly without triangulation
                # We'll try a simple scatter if triangulation is complex, or tricontourf if we assume triangles
                
                try:
                    ax.tricontourf(x, y, depth, levels=20, cmap='viridis_r') # _r for reverse (deep is dark)
                    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis_r'), ax=ax, label='Depth (m)')
                    ax.set_title('Model Domain & Bathymetry')
                except:
                    # Fallback to scatter if triangulation fails (e.g. not enough points or topology issue)
                    ax.scatter(x, y, c=depth, cmap='viridis_r', s=1)
                    ax.set_title('Model Domain & Bathymetry (Scatter)')

                # Save to buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                summary['preview_image'] = img_str
                plt.close(fig)
        except Exception as e:
            summary['preview_error'] = str(e)

    ds.close()
    return summary

def check_compatibility(file1_path, file2_path):
    """
    Checks if two NetCDF files are compatible for direct comparison (subtraction).
    Returns a tuple: (is_compatible: bool, message: str)
    """
    try:
        ds1 = xr.open_dataset(file1_path)
        ds2 = xr.open_dataset(file2_path)
        
        # Check 1: Topology (Node count)
        # SCHISM usually uses 'nSCHISM_hgrid_node'
        node_dim = 'nSCHISM_hgrid_node'
        if node_dim in ds1.dims and node_dim in ds2.dims:
            if ds1.sizes[node_dim] != ds2.sizes[node_dim]:
                return False, f"Grid mismatch: File 1 has {ds1.sizes[node_dim]} nodes, File 2 has {ds2.sizes[node_dim]} nodes."
        else:
            # Fallback to checking all shared dimensions
            common_dims = set(ds1.dims) & set(ds2.dims)
            for dim in common_dims:
                if dim != 'time' and ds1.sizes[dim] != ds2.sizes[dim]:
                     return False, f"Dimension mismatch: '{dim}' differs ({ds1.sizes[dim]} vs {ds2.sizes[dim]})."

        # Check 2: Time Horizon (Optional but good to warn)
        if 'time' in ds1 and 'time' in ds2:
            if ds1.sizes['time'] != ds2.sizes['time']:
                return True, f"Warning: Time steps differ ({ds1.sizes['time']} vs {ds2.sizes['time']}). Comparison will be truncated to the shorter duration."
        
        return True, "Files are compatible."
        
    except Exception as e:
        return False, f"Error checking compatibility: {e}"
    finally:
        if 'ds1' in locals(): ds1.close()
        if 'ds2' in locals(): ds2.close()

if __name__ == "__main__":
    # Test run
    import sys
    if len(sys.argv) > 1:
        print(json.dumps(generate_profile(sys.argv[1]), indent=2))
