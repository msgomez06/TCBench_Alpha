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
from dask.distributed import Client

from utils import toolbox, constants
from utils import data_lib as dlib



# track.plot3D(
#     var="d",
#     timestamps=[track.timestamps[30]],
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
# # %%
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
if __name__ == "__main__":
    print(f"Initializing dask cluster")
    client = Client(n_workers=20, threads_per_worker=1, memory_limit='90GB', processes=True)

    print(f"Reading dataset")
    full_data = toolbox.read_hist_track_file(
        tracks_path="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/tracks/ibtracs/"
    )
    # %%
    data_2005 = full_data[full_data.ISO_TIME.dt.year == 2005]
    katrina = data_2005[data_2005.NAME == "KATRINA"]

    # %%

    ## Inializing dask cluster

    print("Creating track")

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
    track.process_data_collection(
        dc,
        ignore_vars=[
            "specific_cloud_ice_water_content",
            "specific_cloud_liquid_water_content",
            "specific_rain_water_content",
            "specific_snow_water_content",
            # "land_sea_mask",
        ],
        masktype="rect",
        circum_points=30,
    )

    # # %%
    # track.plot_track(
    #     save_path="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/data/track.svg",
    #     track_color="red",
    #     point_color="black",
    #     step=-8,
    # )

    # %%

    track.load_data(ds_type="rect")
    # %%
    track.animate_data(
        "vo",
        ds_type="rect",
        cmap="seismic",
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
    )