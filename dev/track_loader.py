#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:57:33 2023

Script to develop multi-track handling

@author: mgomezd1
"""

#%% Imports

# OS and IO
import os
import sys
import glob
import traceback
import pickle
import matplotlib.pyplot as plt

# Backend Libraries
import numpy as np
import xarray as xr
import pandas as pd

# Retrieve Repository Path
repo_path = '/'+os.path.join(*os.getcwd().split('/')[:-1])

# In order to load functions from scripts located elsewhere in the repository
# it's better to add their path to the list of directories the system will
# look for modules in. We'll add the paths for scripts of interest here.
util_path = f'{repo_path}/utils/'
[sys.path.append(path) for path in [util_path]]

import constants, toolbox
#%% Define the years of interest
years = np.arange(1999, 2000)

#%% Load track data

# Define the path to the track data
track = '/ibtracs'
tracks_path = f'{repo_path}/tracks{track}'

## Define the columns of interest
cols = constants.ibtracs_cols

# Load the track data using the toolbox function
track_data = toolbox.read_hist_track_file(tracks_path = tracks_path,
                                          track_cols = cols)

# Filter the track data to only include the years of interest
track_data = track_data[track_data.ISO_TIME.dt.year.isin(years)]

# Assert that all of the years are present
assert np.all(np.isin(years, track_data.ISO_TIME.dt.year.unique())), 'Not all years are present in the track data'

#%% Data laoding with xarray

# Define the variable of interest
variable = 'mslp'

# Define the path to the data
data_path = f'/work/FAC/FGSE/IDYST/tbeucler/default/saranya/Data/ECMWF/ERA5_25kmx3hr/{variable}/'

# Generate a list of paths to the data
paths = [f'{data_path}*{variable}*{year}*.nc' for year in years]

# Check if each file exists
for idx, path in enumerate(paths):
    assert len(glob.glob(path))==1, f'{variable} file for {years[idx]} not found'

ds = None

#loading arguments for xr.open_mfdataset
kwargs = {'combine':'by_coords',
          'parallel':True,
          'preprocess':None,
          'engine':'netcdf4',
          'chunks':{'time':300,
                  'latitude':100,
                  'longitude':100}
          }

for path in paths:
    print(f'Loading {path}...')
    if ds is None:
        ds = xr.open_mfdataset(path, **kwargs)
    else:
        ds = xr.concat([ds, xr.open_mfdataset(path, **kwargs)], dim='time')
# %%

# Load the keys from the track data metadata
uidx = cols._track_cols__metadata['UID']
name = cols._track_cols__metadata['COSMETIC_NAME']
x = cols._track_cols__metadata['X_coord']
y = cols._track_cols__metadata['Y_coord']
t = cols._track_cols__metadata['TIME_coord']

track_list = []
# Built the track list
for uid in track_data[uidx].unique():
    data = track_data.loc[track_data[uidx]==uid]
    track_list.append(toolbox.tc_track(UID=uid,
                                       NAME=data[name].iloc[0],
                                       track=data[[y,x]].to_numpy(),
                                       timestamps= data[t].to_numpy(),
                                       ))
    
    track_list[-1].add_var_from_dataset(radius=1500,
                                        data = ds,
                                        resolution=0.25,
                                        )

# %%
