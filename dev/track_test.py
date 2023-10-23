#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:57:33 2023

Script to test handling of a single track

@author: mgomezd1
"""

# %% Imports

# OS and IO
import os

import sys
import matplotlib.pyplot as plt

# Backend Libraries
import xarray as xr

from utils import toolbox

# full_data = toolbox.read_hist_track_file()
# %%
data_2005 = full_data[full_data.ISO_TIME.dt.year == 2005]
katrina = data_2005[data_2005.NAME == "KATRINA"]

# %%
track = toolbox.tc_track(
    UID=katrina.SID.iloc[0],
    NAME=katrina.NAME.iloc[0],
    track=katrina[["LAT", "LON"]].to_numpy(),
    timestamps=katrina.ISO_TIME.to_numpy(),
)
# %%
data_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ECMWF/ERA5/" + "mslp/mslp_2005.nc"
)

meteo_data = xr.open_dataset(data_path)

# %%
track.add_var_from_dataset(
    radius=1500,
    data=meteo_data,
    resolution=10,
)

# %%
plot_test = track.rad_ds

vmin = plot_test.var151.min()
vmax = plot_test.var151.max()

for i in range(61):
    plt.figure(dpi=150)
    plot_test.isel(time=i).var151.plot.imshow(vmin=vmin, vmax=vmax)
    plt.show()
    plt.close()
# %%
