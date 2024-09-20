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

full_data = toolbox.read_hist_track_file(
    tracks_path="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/tracks/ibtracs/"
)
# %%
data = full_data[full_data.ISO_TIME.dt.year == 2020]
storm = data[data.NAME == "LAURA"]

# %%
if __name__ == "__main__":
    track = toolbox.tc_track(
        UID=storm.SID.iloc[0],
        NAME=storm.NAME.iloc[0],
        track=storm[["LAT", "LON"]].to_numpy(),
        timestamps=storm.ISO_TIME.to_numpy(),
        ALT_ID=storm[constants.ibtracs_cols._track_cols__metadata.get("ALT_ID")].iloc[
            0
        ],
        wind=storm[constants.ibtracs_cols._track_cols__metadata.get("WIND")].to_numpy(),
        pres=storm[constants.ibtracs_cols._track_cols__metadata.get("PRES")].to_numpy(),
        datadir_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha",
        storm_season=storm.SEASON.iloc[0],
        ai_model="panguweather",
    )

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
