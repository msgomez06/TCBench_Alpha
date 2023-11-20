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

# Backend Libraries
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import joblib as jl

# Local Imports
from utils import constants, toolbox


# Retrieve Repository Path
repo_path = "/" + os.path.join(*os.getcwd().split("/")[:-1])

# %% Define the years of interest
years = np.arange(2020, 2021)

# Define the basin of interest
basin = "NA"

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

# Filter the track data to only include the basin of interest
track_data[track_data["BASIN"] == basin]

# Assert there is at least one track in the basin
assert len(track_data) > 0, f"No tracks found in {basin} basin"
# %%
# Load the keys from the track data metadata
uidx = cols._track_cols__metadata["UID"]
name = cols._track_cols__metadata["COSMETIC_NAME"]
x = cols._track_cols__metadata["X_coord"]
y = cols._track_cols__metadata["Y_coord"]
t = cols._track_cols__metadata["TIME_coord"]

track_list = []

num_trax = len(track_data[uidx].unique())

# Define the variable of interest
variables = [
    "mslp",  # mean sea level pressure
    "uwnd",  # 10m u wind
    "vwnd",  # 10m v wind
    "sst",  # sea surface temperature
    "svars",  # surface variables
    "radvars",  # radiative variables
]
# %%
# Build the track list
for idx, uid in enumerate(track_data[uidx].unique()):
    print("Calculating track", uid, f" {(idx+1)/(num_trax):<5.1%}")

    data = track_data.loc[track_data[uidx] == uid]
    track_list.append(
        toolbox.tc_track(
            UID=uid,
            NAME=data[name].iloc[0],
            track=data[[y, x]].to_numpy(),
            timestamps=data[t].to_numpy(),
        )
    )


# %%
def var_loader(track, variables, years):
    # Data laoding with xarray
    for var in variables:
        # Define the path to the data
        data_path = f"/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ECMWF/ERA5/{var}/"

        # Generate a list of paths to the data
        paths = [f"{data_path}*{var}*{year}*.nc" for year in years]

        # Check if each file exists
        for idx, path in enumerate(paths):
            assert len(glob.glob(path)) == 1, f"{var} file for {years[idx]} not found"

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

        if "lev" in ds.dims:
            ds = ds.isel(lev=0)

        print(f"Adding {var} to track {track.uid} in rect form")
        track.add_var_from_dataset(
            circum_points=20,
            data=ds,
            resolution=0.25,
            masktype="rect",
            num_levels=1 if len(ds.dims) == 3 else ds.dims["plev"],
        )
        print("Done!")
        print(f"Adding {var} to track {track.uid} in rad form")
        track.add_var_from_dataset(
            data=ds,
            resolution=0.25,
            masktype="rad",
            radius=500,
            num_levels=1 if len(ds.dims) == 3 else ds.dims["plev"],
        )
        print("Done!")
        ds.close()


# %% Parallel processing of tracks using joblib
"""
toaster = True

jl.Parallel(
    n_jobs=jl.cpu_count() if toaster else int(jl.cpu_count() / 2),
    verbose=10,
    prefer="threads",
)(jl.delayed(var_loader)(track, variables, years) for track in track_list)
"""

for track in track_list[:2]:
    var_loader(track, variables, years)

# %%
if __name__ == "__main__":
    skip_step = 4
    for track in track_list[:2]:
        # track.rect_ds.isel(time=0).var151.plot.imshow(ax=ax,
        #                                         alpha=0.25,)
        graphDS = (
            track.rect_M_ds.isel(plev=-1).var131 ** 2
            + track.rect_M_ds.isel(plev=-1).var132 ** 2
        ) ** 0.5
        # graphDS = track.rect_S_ds.var34

        vmin = graphDS.min().values
        vmax = graphDS.max().values

        time = pd.to_datetime(graphDS.time.isel(time=0).values).strftime("%Y.%m.%d")

        fig = plt.figure(dpi=300, figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        fig.suptitle("Wind Vector (1000hPa)")
        ax.stock_img()
        ax.set_title(f"{track.name} {time} Rectangular Data")
        for timestamp in graphDS.time[::skip_step]:
            temp_data = graphDS.sel(time=timestamp).plot.imshow(
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
                alpha=0.33,
                add_colorbar=False,
                add_labels=False,
            )

        for timestamp in graphDS.time:
            # select the data that isn't nan
            temp_data = graphDS.sel(time=timestamp).where(
                ~np.isnan(graphDS.sel(time=timestamp)), drop=True
            )

            printtime = pd.to_datetime(timestamp.time.values).strftime("%Y.%m.%d-%H")

            # plot the data as a simple imshow
            new_fig = plt.figure(dpi=300, figsize=(6, 5))
            new_fig.suptitle("Wind Vector (1000hPa)")
            new_ax = plt.axes()
            new_ax.set_box_aspect(1)
            new_ax.set_title(f"{track.name} {printtime}h Rectangular Data")
            temp_data.plot.imshow(
                ax=new_ax,
                vmin=vmin,
                vmax=vmax,
                # transform=ccrs.PlateCarree(),
                alpha=1,
                add_colorbar=True,
                add_labels=False,
            )

        plt.show()
        plt.close()

        """
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
        """

# %%
