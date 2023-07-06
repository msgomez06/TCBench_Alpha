#%% Imports
import xarray as xr
import numpy as np
from os import path, listdir, scandir
import netCDF4 as nc
import sys

# Retrieve Repository Path
repo_path = '/'+os.path.join(*os.getcwd().split('/')[:-1])

# In order to load functions from scripts located elsewhere in the repository
# it's better to add their path to the list of directories the system will
# look for modules in. We'll add the paths for scripts of interest here.
util_path = f'{repo_path}/utils/'
[sys.path.append(path) for path in [util_path]]

import constants

#%% Set up the dataloading paths

base_dir = '/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/CIRA/primed'
years = np.arange(1997, 2020)
basin_set = constants.primed_basins

storm_basins = set()
storm_ids = set()
filepaths = set()
data_fields = set()
for year, basin in [(y, b) for y in years for b in basin_set]:
    try: 
        subdirs = [f.path for f in scandir(f'{base_dir}/{year}/{basin}') if f.is_dir() ]
        print(subdirs)
        for subdir in subdirs:
            filepaths.add(subdir)
            files = listdir(subdir)
            for file in files:
                ds = nc.Dataset(path.join(subdir, file))
                for group in ds.groups.keys():
                    [data_fields.add(var) for var in ds[group].variables]
                #storm_basins.add(ds['overpass_metadata']['basin'][0])
                
                ds.close()
                '''
                if 'era' in file:
                    ds = nc.Dataset(path.join(subdir, file))
                    #print(list(ds['storm_metadata'].variables))
                    
                    storm_basins.add(ds['storm_metadata']['basin'][:])
                    storm_ids.add(ds['storm_metadata']['cyclone_number'])
                    ds.close()
                '''
            #print(storm_basins)          

    except Exception as e:
        print(e)

'''
#file_list = listdir(path.join(base_dir,subdir))
#for file in file_list.copy():
#    if 'era5' not in file:
#        file_list.remove(file)
'''
#%%
# Join the paths to get the dataset path
ds_path = path.join(base_dir,subdir, file_list[0])

# Load the dataset using netcdf4
ds = nc.Dataset(ds_path)

# Store the dataset groups in a dictionary
groups = [*ds.groups.keys()]
# %%
