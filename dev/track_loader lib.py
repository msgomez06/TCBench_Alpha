#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:57:33 2023

Script to develop multi-track handling. 

@author: mgomezd1
"""

# %% Imports

# OS and IO
import os
import sys
import glob
import matplotlib.pyplot as plt

# Backend Libraries
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs

# Retrieve Repository Path
repo_path = "/" + os.path.join(*os.getcwd().split("/")[:-1])

# In order to load functions from scripts located elsewhere in the repository
# it's better to add their path to the list of directories the system will
# look for modules in. We'll add the paths for scripts of interest here.
util_path = f"{repo_path}/utils/"
[sys.path.append(path) for path in [util_path]]

from utils import constants, toolbox

# %% Define the years of interest
years = np.arange(2020, 2021)

# %% Load track data

# Define the path to the track data
track = "/ibtracs"
tracks_path = f"{repo_path}/tracks{track}"

## Define the columns of interest
cols = constants.ibtracs_cols

# Load the track data using the toolbox function
track_data = toolbox.read_hist_track_file(
    tracks_path=tracks_path,
    track_cols=cols,
    backend=cols._track_cols__metadata["loader"],
)

t = cols._track_cols__metadata["TIME_coord"]

# Filter the track data to only include the years of interest
track_data = track_data[track_data[t].dt.year.isin(years)]

# Assert that all of the years are present
assert np.all(
    np.isin(years, track_data[t].dt.year.unique())
), "Not all years are present in the track data"

# %% Data laoding with xarray

# Define the variable of interest
variable = "mslp"

# Define the path to the data
data_path = f"/work/FAC/FGSE/IDYST/tbeucler/default/saranya/Data/ECMWF/ERA5_25kmx3hr/{variable}/"

# Generate a list of paths to the data
paths = [f"{data_path}*{variable}*{year}*.nc" for year in years]

# Check if each file exists
for idx, path in enumerate(paths):
    assert len(glob.glob(path)) == 1, f"{variable} file for {years[idx]} not found"

ds = None

# loading arguments for xr.open_mfdataset
kwargs = {
    "combine": "by_coords",
    "parallel": True,
    "preprocess": None,
    "engine": "netcdf4",
    "chunks": {"time": 300, "latitude": 100, "longitude": 100},
}

for path in paths:
    print(f"Loading {path}...")
    if ds is None:
        ds = xr.open_mfdataset(path, **kwargs)
    else:
        ds = xr.concat([ds, xr.open_mfdataset(path, **kwargs)], dim="time")
# %%
# Load the keys from the track data metadata
uidx = cols._track_cols__metadata["UID"]
name = cols._track_cols__metadata["COSMETIC_NAME"]
x = cols._track_cols__metadata["X_coord"]
y = cols._track_cols__metadata["Y_coord"]
t = cols._track_cols__metadata["TIME_coord"]

track_list = []

num_trax = len(track_data[uidx].unique())

# Build the track list
for idx, uid in enumerate(track_data[uidx].unique()):
    print("\r Calculating track", uid, f" {(idx+1)/(num_trax):<5.1%}", end="")
    data = track_data.loc[track_data[uidx] == uid]
    track_list.append(
        toolbox.tc_track(
            UID=uid,
            NAME=data[name].iloc[0],
            track=data[[y, x]].to_numpy(),
            timestamps=data[t].to_numpy(),
        )
    )

    track_list[-1].add_var_from_dataset(
        circum_points=5, data=ds, resolution=1, masktype="rect"
    )

    track_list[-1].add_var_from_dataset(
        data=ds, resolution=1, masktype="rad", radius=500
    )

# %%
if __name__ == "__main__":
    skip_step = 4
    for track in track_list:
        # track.rect_ds.isel(time=0).var151.plot.imshow(ax=ax,
        #                                         alpha=0.25,)
        vmin = track.rad_ds.var151.min().values
        vmax = track.rad_ds.var151.max().values

        time = pd.to_datetime(track.rect_ds.time.isel(time=0).values).strftime(
            "%Y.%m.%d"
        )

        fig = plt.figure(dpi=300)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_title("Mean Sea Level Pressure")
        ax.coastlines()
        fig.suptitle(f"{track.name} {time} Radial Data")
        for timestamp in track.rad_ds.time[::skip_step]:
            temp_data = track.rad_ds.sel(time=timestamp).var151.plot.imshow(
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
                alpha=0.33,
                add_colorbar=False,
            )

        plt.show()
        plt.close()

        fig = plt.figure(dpi=300)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_title("Mean Sea Level Pressure")
        ax.coastlines()
        fig.suptitle(f"{track.name} {time} Rectangular Data")
        for timestamp in track.rect_ds.time[::skip_step]:
            temp_data = track.rect_ds.sel(time=timestamp).var151.plot.imshow(
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
                alpha=0.33,
                add_colorbar=False,
            )

        plt.show()
        plt.close()


# %%
