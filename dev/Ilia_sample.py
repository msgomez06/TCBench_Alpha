# %% This line makes it run on jupyter notebook by default in vscode
#  Imports
# OS and IO
import os
import sys
import pickle
import subprocess
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dask.array as da
from sklearn.preprocessing import StandardScaler
import torch
import dask
import multiprocessing
import pandas as pd

# from dask import optimize
import time


# Backend Libraries
import joblib as jl

from utils import toolbox, ML_functions as mlf
import metrics

print("Imports successful", flush=True)

#  Setup
datadir = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha"
cache_dir = "/scratch/mgomezd1/cache"
result_dir = (
    "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/"
)
# %%
tracks = toolbox.get_TC_seasons(
    season_list=list(np.arange(2020, 2021)), datadir_path=datadir
)

for season, storm_list in tracks.items():
    for storm in storm_list.copy():
        if storm.ALT_ID is None or not ("AL" in storm.ALT_ID):
            storm_list.remove(storm)

storms = tracks[2020][:3]


headers = toolbox.constants.default_ships_vars

# %%
for storm in storms:
    storm.ReAnal.animate_2d()

# %%
