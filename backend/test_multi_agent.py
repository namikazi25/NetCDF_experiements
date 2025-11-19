import os
import xarray as xr
import numpy as np
from orchestrator import run_orchestrator

def create_dummy_nc(filename="test_multi_agent.nc"):
    data = xr.DataArray(np.random.randn(10, 10), dims=("x", "y"), coords={"x": range(10), "y": range(10)})
    ds = xr.Dataset({"temperature": data})
    ds.attrs["description"] = "Test dataset"
    ds.to_netcdf(filename)
    return filename

def test_orchestrator():
    print("Creating dummy NetCDF file...")
    filename = create_dummy_nc()
    
    try:
        print("Testing Orchestrator...")
        metadata = {
            "dims": {"x": 10, "y": 10},
            "data_vars": {"temperature": {"dims": ("x", "y"), "dtype": "float64"}}
        }
        
        query = "Calculate the mean temperature."
        
        result = run_orchestrator(query, metadata, filename)
        print("Steps Executed:")
        for step in result["steps_log"]:
            print(f"- {step['stage']}: {step['status']}")
            
        print("\nFinal Response Preview:", result["response"][:100])
        
    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_orchestrator()
