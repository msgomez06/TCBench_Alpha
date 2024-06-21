# %%
import numpy as np
import pandas as pd
import os
import xarray as xr

from utils import toolbox, constants

# To make this notebook's output stable across runs
rnd_seed = 42
rnd_gen = np.random.default_rng(rnd_seed)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

# %%
# We'll load 10 seasons, from 2010 - 2019, for this exercise
tracks = toolbox.get_TC_seasons(
    season_list=[
        *range(2016, 2020),
    ],
    datadir_path="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/data/",
)
# %%
# We'll define the number of hours into the future we'll be predicting
delta_target = 24

# and store the minimum and maximum leadtimes (both = 0)
leadtime_min = 0
leadtime_max = 0

# And define the number of steps for each predictor.
timeseries_length = 5

# Let's define the splits dictionary
splits = {"train": 0.6, "validation": 0.2, "test": 0.2}

# And finally, the location for the data folder
timeseries_dir = (
    "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/data/SHIPS_netcdfs"
)

data_dictionary = toolbox.get_timeseries_sets(
    splits=splits,
    season_dict=tracks,
    leadtime_min=leadtime_min,
    leadtime_max=leadtime_max,
    timeseries_length=timeseries_length,
    delta_target=delta_target,
    timeseries_dir=timeseries_dir,
    test_strategy="random",
)

# %%
