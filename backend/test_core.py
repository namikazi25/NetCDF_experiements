import xarray as xr
import numpy as np
import os
from nc_processor import extract_metadata
from llm_service import chat_with_context

def create_dummy_nc(filename="test.nc"):
    data = xr.DataArray(np.random.randn(2, 3), dims=("x", "y"), coords={"x": [10, 20], "y": [10, 20, 30]})
    ds = xr.Dataset({"temperature": data})
    ds.attrs["description"] = "Test dataset"
    ds.to_netcdf(filename)
    return filename

def test_pipeline():
    print("Creating dummy NetCDF file...")
    filename = create_dummy_nc()
    
    try:
        print("Extracting metadata...")
        metadata = extract_metadata(filename)
        print("Metadata extracted:", metadata.keys())
        
        # Test multi-file structure
        combined_metadata = {"files": {filename: metadata}}
        
        print("Testing LLM context generation (mocking LLM call for safety)...")
        # We won't actually call the LLM here to avoid API costs/errors in test, 
        # but we can check if the context formatting works.
        from llm_service import format_metadata_context
        context = format_metadata_context(combined_metadata)
        print("Context generated successfully.")
        print(context[:100] + "...")
        
        print("Verification successful!")
    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_pipeline()
