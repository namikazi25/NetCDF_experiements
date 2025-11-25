
import xarray as xr

def check_shapes():
    path = "uploads/schouts_2.nc"
    ds = xr.open_dataset(path)
    
    print("Sizes:", ds.sizes)
    # Check if elements dim exists
    if 'SCHISM_hgrid_face_nodes' in ds:
        print("Elements:", ds.dims.get('nSCHISM_hgrid_face', 'Unknown'))
        
    for var in ['depth', 'elev', 'wsh_x', 'wsh_y', 'hvel_x']:
        if var in ds:
            print(f"{var}: {ds[var].shape} dims: {ds[var].dims}")
        else:
            print(f"{var}: Not found")
            
    ds.close()

if __name__ == "__main__":
    check_shapes()
