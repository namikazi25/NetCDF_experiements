import xarray as xr
import numpy as np

def convert_to_serializable(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def extract_metadata(file_path: str) -> dict:
    """
    Opens a NetCDF file and extracts metadata about variables, dimensions, and attributes.
    """
    try:
        ds = xr.open_dataset(file_path)
        
        metadata = {
            "dims": dict(ds.sizes),
            "coords": list(ds.coords.keys()),
            "data_vars": {},
            "attrs": ds.attrs
        }
        
        for var_name, da in ds.data_vars.items():
            metadata["data_vars"][var_name] = {
                "dims": da.dims,
                "attrs": da.attrs,
                "dtype": str(da.dtype),
                "shape": da.shape
            }
            
        ds.close()
        return convert_to_serializable(metadata)
    except Exception as e:
        raise Exception(f"Failed to process NetCDF file: {str(e)}")
