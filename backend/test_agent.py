import os
import xarray as xr
import numpy as np
from agent_workflow import run_agent_workflow

def create_dummy_nc(filename="test_agent.nc"):
    data = xr.DataArray(np.random.randn(10, 10), dims=("x", "y"), coords={"x": range(10), "y": range(10)})
    ds = xr.Dataset({"temperature": data})
    ds.attrs["description"] = "Test dataset"
    ds.to_netcdf(filename)
    return filename

def test_agent():
    print("Creating dummy NetCDF file...")
    filename = create_dummy_nc()
    
    try:
        print("Testing Agent Workflow...")
        metadata = {
            "dims": {"x": 10, "y": 10},
            "data_vars": {"temperature": {"dims": ("x", "y"), "dtype": "float64"}}
        }
        
        # Mock query
        query = "Calculate the mean temperature."
        
        # We are testing if the code runs without crashing. 
        # The LLM call might fail if no API key, but we want to see if the structure holds.
        # If API key is missing, it will return an error message, which is fine for this test.
        
        result = run_agent_workflow(query, metadata, filename)
        print("Result:", result)
        
    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_agent()
