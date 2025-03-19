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
import pandas as pd
import numpy as np
from importlib import reload

# Backend Libraries
import xarray as xr

from utils import toolbox, constants
from utils.toolbox import *
from utils import data_lib as dlib

# import cartopy for coastlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature

full_data = toolbox.read_hist_track_file(
    tracks_path="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/tracks/ibtracs/"
)
results_dir = (
    "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/"
)

# %%
if __name__ == "__main__":
    initial_times = []
    for name, year in [
        ("DORIAN", 2019),
        ("KATRINA", 2005),
        ("HAITANG", 2005),
        ("HARVEY", 2017),
        ("TITLI", 2018),
        ("VARDAH", 2016),
        ("MARIA", 2017),
        ("WILMA", 2011),
        ("HAIYAN", 2013),
        ("PAM", 2015),
        ("ENAWO", 2017),
        ("KENNETH", 2011),
    ]:
        print(f"Processing {name} {year}")
        data = full_data[full_data.ISO_TIME.dt.year == year]
        storm = data[data.NAME == name]
        track = toolbox.tc_track(
            UID=storm.SID.iloc[0],
            NAME=storm.NAME.iloc[0],
            track=storm[["LAT", "LON"]].to_numpy(),
            timestamps=storm.ISO_TIME.to_numpy(),
            ALT_ID=storm[
                constants.ibtracs_cols._track_cols__metadata.get("ALT_ID")
            ].iloc[0],
            wind=storm[
                constants.ibtracs_cols._track_cols__metadata.get("WIND")
            ].to_numpy(),
            pres=storm[
                constants.ibtracs_cols._track_cols__metadata.get("PRES")
            ].to_numpy(),
            datadir_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha",
            storm_season=storm.SEASON.iloc[0],
            ai_model="panguweather",
        )

        times = pd.to_datetime(track.timestamps)
        valid_times = times[np.isin(times.hour, [0, 6, 12, 18])]

        start_date = valid_times[0]
        end_date = valid_times[-1]

        num_steps = (
            (end_date - start_date).to_numpy() / np.timedelta64(6, "h")
        ).astype(int)
        leadtimes = np.arange(
            -4 * 5 * 6,  # Start 5 days before
            num_steps * 6,  # End at the number of steps * 6 hours
            6,  # 6 hour intervals
        ).astype("timedelta64[h]")
        dates = start_date.to_numpy() + leadtimes

        # Generate 6-hourly leadtimes between the start and end dates

        # # Generate the leadtimes up to 5 days before with 6 hour intervals
        # leadtimes = np.arange(-5 * 24, 0, 6).astype("timedelta64[h]")
        # genesis_dates = start_date.to_numpy() + leadtimes
        # genesis_dates = genesis_dates.astype("datetime64[ns]")

        # valid_times = valid_times.to_numpy()

        # valid_times = np.hstack([genesis_dates, valid_times])
        # valid_times = np.unique(valid_times)

        # # Check that the valid times are continuous
        # assert np.all(
        #     np.diff(valid_times).astype("timedelta64[h]") == np.timedelta64(6, "h")
        # )

        # print(valid_times.shape)

        initial_times.append(dates)

    unique_times = np.unique(np.concatenate(initial_times))
    np.save(f"{results_dir}Initial_Times.npy", unique_times)

    # %% Abstract Image

    # track.ReAnal.load()
    # track.ai.load()
    # era5 = track.ReAnal.ds
    # pangu = track.ai.ds

    # era5["wind"] = (era5["10u"] ** 2 + era5["10v"] ** 2) ** 0.5
    # pangu["wind"] = (pangu["u10"] ** 2 + pangu["v10"] ** 2) ** 0.5

    # forecast = pangu.sel(leadtime_hours=96)

    # snapshot = forecast.isel(time=10).wind

    # target_time = snapshot.time.values + np.timedelta64(96, "h")
    # era5_snapshot = era5.wind.sel(time=target_time)
    # era5_IC = era5.sel(time=snapshot.time.values).wind

    # delta = snapshot - era5_snapshot

    # cropped_snapshot = snapshot.where(delta.notnull(), drop=True)
    # delta = delta.where(delta.notnull(), drop=True)
    # era5_snapshot = era5_snapshot.where(delta.notnull(), drop=True)
    # era5_IC = era5_IC.where(era5_IC.notnull(), drop=True)

    # print_time = pd.Timestamp(target_time).strftime("%Y-%m-%d %H:%M")

    # min_val = min(
    #     cropped_snapshot.min().values,
    #     era5_snapshot.min().values,
    #     era5_IC.min().values,
    #     # delta.min().values,
    # )

    # max_val = max(
    #     cropped_snapshot.max().values,
    #     era5_snapshot.max().values,
    #     era5_IC.max().values,
    #     # delta.max().values,
    # )

    # Select the non-nan areas
    # %%
    # create a figure and axis
    fig, axs = plt.subplots(
        2, 2, subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 10)
    )

    cmap = "RdBu"
    cmap2 = "PiYG"

    title_dict = {
        "fontsize": "xx-large",
        "fontweight": mpl.rcParams["axes.titleweight"],
        "verticalalignment": "baseline",
        "horizontalalignment": "center",
    }

    # plot the IC
    era5_IC.plot.pcolormesh(
        ax=axs[0, 0],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        add_colorbar=False,
        vmin=min_val,
        vmax=max_val,
    )
    axs[0, 0].coastlines()
    axs[0, 0].set_title(f"ERA5 IC ({print_time})", fontdict=title_dict)
    gl = axs[0, 0].gridlines(draw_labels=True, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 12, "color": "white"}
    gl.ylabel_style = {"size": 12, "color": "white"}

    # plot the forecast
    cropped_snapshot.plot.pcolormesh(
        ax=axs[0, 1],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        add_colorbar=False,
        vmin=min_val,
        vmax=max_val,
    )
    axs[0, 1].coastlines()
    axs[0, 1].set_title("PanguWeather 96h Forecast", fontdict=title_dict)
    gl2 = axs[0, 1].gridlines(draw_labels=True, linestyle="--")
    gl2.top_labels = False
    gl2.right_labels = False
    gl2.xlabel_style = {"size": 12, "color": "white"}
    gl2.ylabel_style = {"size": 12, "color": "white"}

    # plot the truth
    era5_snapshot.plot.pcolormesh(
        ax=axs[1, 0],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        add_colorbar=False,
        vmin=min_val,
        vmax=max_val,
    )
    axs[1, 0].coastlines()
    axs[1, 0].set_title("ERA5 at t+96h", fontdict=title_dict)
    gl3 = axs[1, 0].gridlines(draw_labels=True, linestyle="--")
    gl3.top_labels = False
    gl3.right_labels = False
    gl3.xlabel_style = {"size": 12, "color": "white"}
    gl3.ylabel_style = {"size": 12, "color": "white"}

    # plot the difference
    delta.plot.pcolormesh(
        ax=axs[1, 1],
        transform=ccrs.PlateCarree(),
        cmap=cmap2,
        add_colorbar=False,
        vmin=delta.min().values,
        vmax=delta.max().values,
    )
    axs[1, 1].coastlines()
    axs[1, 1].set_title(
        f"Difference, range: [{delta.min().values:.2f} , {delta.max().values:.2f} m/s]",
        fontdict=title_dict,
    )
    gl4 = axs[1, 1].gridlines(draw_labels=True, linestyle="--")
    gl4.top_labels = False
    gl4.right_labels = False
    gl4.xlabel_style = {"size": 12, "color": "white"}
    gl4.ylabel_style = {"size": 12, "color": "white"}

    fig.suptitle(f"Hurricane Dorian (2019)", fontsize="xx-large", color="white")

    # Add a colorbar to the right of the plot, with white ticks and labels
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig.colorbar(sm, cax=cbar_ax, label="Wind Speed (m/s)")

    # change the colorbar ticks to white
    cbar_ax.yaxis.set_tick_params(color="white")
    cbar_ax.yaxis.set_tick_params(color="white")

    # change the colorbar label to white
    cbar_ax.yaxis.label.set_color("white")

    # change the colorbar ticklabels to white
    plt.setp(plt.getp(cbar_ax.axes, "yticklabels"), color="white")

    toolbox.plot_facecolors(fig=fig, axes=axs)

    # %%
    if False:
        track.ai.load()
        track.ai.ds.z500.attrs["longname"] = "500 hPa Geopotential Height"
        track.ai.ds.t850.attrs["longname"] = "850 hPa Temperature"
        msl_attrs = track.ai.ds.msl.attrs

        # convert msl to hPa
        track.ai.ds["msl"] = track.ai.ds.msl / 100
        track.ai.ds.msl.attrs = msl_attrs

        track.ai.animate_var("u10", plot_kwargs={"cmap": "seismic"})
        track.ai.animate_var("msl", plot_kwargs={"cmap": "twilight"})
        track.ai.animate_var("v10", plot_kwargs={"cmap": "seismic"})
        track.ai.animate_var("z500", plot_kwargs={"cmap": "twilight"})
        track.ai.animate_var("t850", plot_kwargs={"cmap": "seismic"})

    # track.filepath = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha/2019/"

    # track.load_ai_data()
    # %%
    # calculate the wind from the components in AI_ds and store it in the dataset
    # wind = (track.AI_ds.u10**2 + track.AI_ds.v10**2) ** 0.5
    # wind.attrs["units"] = "m/s"
    # wind.attrs["longname"] = "Wind Speed"
    # track.AI_ds["wind_speed"] = wind
    # msl_attrs = track.AI_ds.msl.attrs
    # track.AI_ds["msl"] = track.AI_ds.msl / 100
    # msl_attrs["units"] = "hPa"
    # track.AI_ds.msl.attrs = msl_attrs

    # track.animate_AI_ds(var="msl", plot_kwargs={"cmap": "twilight"})
    # # track.animate_AI_ds(var="wind_speed", plot_kwargs={"cmap": "seismic"})
    # # %%
    # data_dir = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ECMWF/ERA5/"
    # save_path = "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/data/"

    # dc = dlib.Data_Collection(data_dir)

    # # %%

    # track.process_data_collection(
    #     dc,
    #     reanal_variables=[
    #         "10m_u_component_of_wind",
    #         "10m_v_component_of_wind",
    #         "mean_sea_level_pressure",
    #         "temperature",
    #         "geopotential",
    #     ],
    #     masktype="rect",
    #     circum_points=30 * 4,
    #     plevels={"temperature": [850], "geopotential": [500]},
    #     verbose=True,
    #     n_jobs=4,
    #     # n_workers=16,
    #     # mem_limit="40GB",
    # )

# %%
