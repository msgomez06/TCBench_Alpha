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
import matplotlib as mpl

# Backend Libraries
import xarray as xr

from utils import toolbox, constants
from utils import data_lib as dlib

full_data = toolbox.read_hist_track_file(
    tracks_path="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/tracks/ibtracs/"
)
# %%
data_2005 = full_data[full_data.ISO_TIME.dt.year == 2020]
storm = data_2005[data_2005.NAME == "LAURA"]

# %%
track = toolbox.tc_track(
    UID=storm.SID.iloc[0],
    NAME=storm.NAME.iloc[0],
    track=storm[["LAT", "LON"]].to_numpy(),
    timestamps=storm.ISO_TIME.to_numpy(),
    ALT_ID=storm[constants.ibtracs_cols._track_cols__metadata.get("ALT_ID")].iloc[0],
    wind=storm[constants.ibtracs_cols._track_cols__metadata.get("WIND")].to_numpy(),
    pres=storm[constants.ibtracs_cols._track_cols__metadata.get("PRES")].to_numpy(),
)
# %%
data_dir = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ECMWF/ERA5/"
save_path = "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/data/"

# dc = dlib.Data_Collection(data_dir)

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

# # %%
# track.plot_track(
#     save_path="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/data/track.svg",
#     track_color="red",
#     point_color="black",
#     step=-8,
# )

# %%
# dc = dlib.AI_Data_Collection(
#     "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/AI-milton/panguweather"
# )

# tst = track.process_data_collection(dc)

# track.load_data(ds_type="rect")
# track.load_timeseries()

# # %%
# wind_speed = (track.rect_ds.u**2 + track.rect_ds.v**2) ** 0.5
# wind_speed.attrs["units"] = "m/s"
# wind_speed.attrs["long_name"] = "Wind Speed"
# track.rect_ds["wind_speed"] = wind_speed

# # %%
# track.animate_data(
#     "vo",
#     ds_type="rect",
#     cmap="cividis",
#     ignore_levels=[
#         150,
#         70,
#         50,
#         30,
#         20,
#         10,
#     ],
# )
# # %%
# track.plot3D(
#     var="vo",
#     timestamp=track.timestamps[30],
#     ds_type="rect",
#     alpha=0.25,
#     ignore_levels=[
#         # 1000,
#         # 925,
#         # 850,
#         # 700,
#         # 600,
#         # 500,
#         # 400,
#         # 300,
#         # 200,
#         150,
#         70,
#         50,
#         30,
#         20,
#         10,
#     ],
#     figsize=(4, 6),
#     cmap="seismic",
#     # facecolor="white",
#     # text_color="black",
# )
# %%
# track.rect_ds.isel(time=30).where(
#     ~track.rect_ds.isel(time=30).isnull(), drop=True
# ).msl.plot.imshow(cmap="winter_r")

# # %%
# fig, ax = plt.subplots(figsize=(4, 6), dpi=300)
# plot_data = track.rect_ds.isel(time=30)
# image = plot_data.where(~plot_data.isnull(), drop=True).msl.plot.imshow(
#     ax=ax, add_colorbar=False, cmap="viridis_r"
# )
# fig.set_facecolor((0.3, 0.3, 0.3))
# ax.set_facecolor((0.3, 0.3, 0.3))
# ax.spines["top"].set_color("white")
# ax.spines["right"].set_color("white")
# ax.spines["bottom"].set_color("white")
# ax.spines["left"].set_color("white")
# ax.tick_params(axis="x", colors="white")
# ax.tick_params(axis="y", colors="white")
# ax.title.set_color("white")
# ax.yaxis.label.set_color("white")
# ax.xaxis.label.set_color("white")
# cbar = fig.colorbar(
#     image, ax=ax, label="Mean Sea Level Pressure", orientation="horizontal"
# )
# cbar.ax.xaxis.label.set_color("white")
# cbar.ax.tick_params(axis="x", colors="white", labelsize=8)
# cbar.ax.spines["top"].set_color("white")
# cbar.ax.spines["right"].set_color("white")
# cbar.ax.spines["bottom"].set_color("white")
# plt.show()

# # %%
# fig, ax = plt.subplots(figsize=(4, 6), dpi=300)
# valid_data = plot_data.where(~plot_data.isnull(), drop=True)
# wind = (valid_data.u10**2 + valid_data.v10**2) ** 0.5
# wind.attrs["units"] = "m/s"
# wind.attrs["long_name"] = "2m Wind Speed"
# image = wind.plot.imshow(cmap="plasma", add_colorbar=False)
# fig.set_facecolor((0.3, 0.3, 0.3))
# ax.set_facecolor((0.3, 0.3, 0.3))
# ax.spines["top"].set_color("white")
# ax.spines["right"].set_color("white")
# ax.spines["bottom"].set_color("white")
# ax.spines["left"].set_color("white")
# ax.tick_params(axis="x", colors="white")
# ax.tick_params(axis="y", colors="white")
# ax.title.set_color("white")
# ax.yaxis.label.set_color("white")
# ax.xaxis.label.set_color("white")
# cbar = fig.colorbar(image, ax=ax, label="2m Wind Speed [m/s]", orientation="horizontal")
# cbar.ax.xaxis.label.set_color("white")
# cbar.ax.tick_params(axis="x", colors="white", labelsize=8)
# cbar.ax.spines["top"].set_color("white")
# cbar.ax.spines["right"].set_color("white")
# cbar.ax.spines["bottom"].set_color("white")
# plt.show()
# # %%

# fig, axes = plt.subplots(2, 1, figsize=(10, 4), dpi=200)
# fig.set_facecolor((0.3, 0.3, 0.3))


# min_pressure = track.rect_ds.msl.min(dim=["latitude", "longitude"])
# pressure_plot = min_pressure.plot(ax=axes[0],
#                                   color = mpl.colors.TABLEAU_COLORS["tab:pink"])

# max_wind = (track.rect_ds.u10**2 + track.rect_ds.v10**2) ** 0.5
# max_wind.attrs["units"] = "m/s"
# max_wind.attrs["long_name"] = "Maximum 2m Wind Speed"
# max_wind.attrs["standard_name"] = "max_wind"
# wind_plot = max_wind.max(dim=["latitude", "longitude"]).plot(
#     ax=axes[1], label="Max 2m Wind Speed",
#     color=mpl.colors.TABLEAU_COLORS["tab:green"]
# )

# axes[0].set_ylabel("MSLP Min [Pa]")
# axes[1].set_ylabel("2m Wind Speed Max [m/s]")

# for ax in axes:
#     ax.set_xlabel("Date")
#     ax.set_facecolor((0.3, 0.3, 0.3))
#     ax.spines["top"].set_color("white")
#     ax.spines["right"].set_color("white")
#     ax.spines["bottom"].set_color("white")
#     ax.spines["left"].set_color("white")
#     ax.tick_params(axis="x", colors="white")
#     ax.tick_params(axis="y", colors="white")
#     ax.title.set_color("white")
#     ax.yaxis.label.set_color("white")
#     ax.xaxis.label.set_color("white")
# fig.tight_layout()

# # %%

# %%
# import xarray as xr
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# ds = xr.open_dataset('/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/data/2020233N14313.AI.panguweather.nc')


# for idx in range(ds.leadtime_hours.size):
#     plt.rcParams['font.size'] = 16
#     test = ds.isel({'time':10, 'leadtime_hours':idx})
#     wind=((test.u10 **2 + test.v10**2)**.5)
#     wind.attrs = {'long_name':'10m Wind Magnitude', 'units':'m s**-1'}
#     wind = wind.where(~wind.isnull(), drop=True)
#     fig,ax = plt.subplots(dpi=300, figsize=(8,8))
#     ax.set_aspect(1)
#     im = wind.plot.imshow(cmap='magma', ax=ax, add_colorbar=False)

#     # Create a divider for the existing axes instance
#     divider = make_axes_locatable(ax)

#     # Append axes to the right of ax, with 5% width of ax
#     cax = divider.append_axes("right", size="5%", pad=0.1)

#     # Create colorbar in the appended axes
#     # Tick locations can be set with the `ticks` keyword
#     fig.colorbar(im, cax=cax, label=f"{wind.attrs['long_name']} ({wind.attrs['units']}")

#     ax.set_title(f"Date:{wind.time.dt.strftime('%y.%m.%d-%H:00').values.item()} Lead: {wind.leadtime_hours.values.item()}h")
#     fig.savefig(f"Laura {wind.time.dt.strftime('%y.%m.%d-%Hh00').values.item()}_{wind.leadtime_hours.values.item()}h.png", transparent=True)
#     plt.close()
