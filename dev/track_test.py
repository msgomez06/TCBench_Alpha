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

from utils import toolbox, constants
from utils import data_lib as dlib

full_data = toolbox.read_hist_track_file(
    tracks_path="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/tracks/ibtracs/"
)
# %%
data_2005 = full_data[full_data.ISO_TIME.dt.year == 2005]
katrina = data_2005[data_2005.NAME == "KATRINA"]

# %%
track = toolbox.tc_track(
    UID=katrina.SID.iloc[0],
    NAME=katrina.NAME.iloc[0],
    track=katrina[["LAT", "LON"]].to_numpy(),
    timestamps=katrina.ISO_TIME.to_numpy(),
    ALT_ID=katrina[constants.ibtracs_cols._track_cols__metadata.get("ALT_ID")].iloc[0],
)
# %%
data_dir = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ECMWF/ERA5/"
# data_path = data_dir + "legacy_files/gpot/gpot_2005.nc"

dc = dlib.Data_Collection(data_dir)
# %%
# track.process_data_collection(
#     dc,
#     ignore_vars=[
#         "specific_cloud_ice_water_content",
#         "specific_cloud_liquid_water_content",
#         "specific_rain_water_content",
#         "specific_snow_water_content",
#         # "land_sea_mask",
#     ],
#     masktype="rect",
#     circum_points=30,
# )

# %%
track.plot_track()

# %%

track.load_data(ds_type="rect")
# %%
track.plot3D(
    var="q",
    timestamps=[track.timestamps[30]],
    ds_type="rect",
    alpha=0.25,
    ignore_levels=[
        # 1000,
        # 925,
        # 850,
        # 700,
        # 600,
        # 500,
        # 400,
        # 300,
        # 200,
        150,
        70,
        50,
        30,
        20,
        10,
    ],
    figsize=(4, 6),
)

# # %%

# %%
