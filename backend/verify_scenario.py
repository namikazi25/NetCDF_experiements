
import xarray as xr
import os

def verify_scenario():
    path = "uploads/schouts_2.nc"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        ds = xr.open_dataset(path)
        print("Successfully opened scenario file.")
        print("Variables:", list(ds.data_vars))
        print("Coords:", list(ds.coords))
        ds.close()
    except Exception as e:
        print(f"Failed to open file: {e}")

if __name__ == "__main__":
    verify_scenario()
