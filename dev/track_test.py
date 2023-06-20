#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:57:33 2023

Script to test handling of a single track

@author: mgomezd1
"""

#%% Imports

# OS and IO
import os
import sys
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


full_data = toolbox.read_hist_track_file()
#%%
data_2005 = full_data[full_data.ISO_TIME.dt.year == 2005]
katrina = data_2005[data_2005.NAME == 'KATRINA']

track = toolbox.tc_track(UID=katrina.SID.iloc[0],
                        NAME=katrina.NAME.iloc[0],
                        track=katrina[['LAT','LON']].to_numpy(),
                        timestamps= katrina.ISO_TIME.to_numpy(),
                        )
#%%
data_path = '/work/FAC/FGSE/IDYST/tbeucler/default/saranya/Data/ECMWF/ERA5_25kmx3hr/mslp/mslp_2005.nc'

meteo_data = xr.open_dataset(data_path)

#%%
track.add_var_from_dataset(radius=10000,
                           data = meteo_data,
                           resolution=5,
                           )

#%%
plot_test = track.ds

vmin = plot_test.var151.min()
vmax = plot_test.var151.max()

for i in range(61):
    plt.figure(dpi=150)
    plot_test.isel(time=i).var151.plot.imshow(vmin=vmin, vmax=vmax)
    plt.show()
    plt.close()
# %%
