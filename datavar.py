import xarray as xr

# Open the NetCDF file
ds = xr.open_dataset(r'C:\Users\harsh.PC\Downloads\ML project\combined_output.nc')

# List only the data variables
print(list(ds.data_vars))

# List all variables (including coordinates like lat, lon, time)
print(list(ds.variables))