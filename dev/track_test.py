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

from utils import toolbox, constants, data_lib as dlib

full_data = toolbox.read_hist_track_file()
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
#         "land_sea_mask",
#     ],
#     masktype="rect",
#     circum_points=20,
# )

track.load_data(ds_type="rect")


# meteo_data = xr.open_dataset(data_path).isel(plev=0)
# # %%
# # track.add_var_from_dataset(
# #     radius=500,
# #     data=meteo_data,
# #     resolution=1,
# #     num_levels=1 if len(meteo_data.dims) == 3 else meteo_data.dims["plev"],
# # )

# # %%
# track.add_var_from_dataset(
#     circum_points=20,
#     data=meteo_data,
#     resolution=0.25,
#     masktype="rect",
#     num_levels=1 if len(meteo_data.dims) == 3 else meteo_data.dims["plev"],
# )

# # %%
# plot_test = track.rect_S_ds

# vmin = plot_test.var129.min()
# vmax = plot_test.var129.max()
# plot_test = plot_test.where(~plot_test.isnull(), drop=True)
# for i in range(0, 61, 5):
#     plt.figure(dpi=150)
#     plt.imshow(
#         plot_test.isel(time=i)
#         .where(~plot_test.isel(time=i).isnull(), drop=True)
#         .var129.values,
#         vmin=vmin,
#         vmax=vmax,
#     )
#     # plot_test.isel(time=i).var129.plot.imshow(vmin=vmin, vmax=vmax)
#     plt.show()
#     plt.close()
# # %%
# vmin = plot_test.var34.min()
# vmax = plot_test.var34.max()

# for i in range(0, 61):
#     plt.figure(dpi=150)
#     plot_test.isel(time=i).var34.plot.imshow(vmin=vmin, vmax=vmax)
#     plt.show()
#     plt.close()
# # %%

# %%
