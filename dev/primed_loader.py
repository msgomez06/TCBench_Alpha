#%%
import xarray as xr
from os import path, listdir
import netCDF4 as nc

base_dir = '/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/CIRA/primed/'
subdir = '2020/AL/01/'

file_list = listdir(path.join(base_dir,subdir))
for file in file_list.copy():
    if 'era5' not in file:
        file_list.remove(file)
#%%
ds_path = path.join(base_dir,subdir, file_list[0])
ds = nc.Dataset(ds_path)

print(ds.groups)

#%%


test_data = xr.open_dataset(path.join(base_dir,subdir, file_list[0]),
                            group = 'rectilinear')


# %%
