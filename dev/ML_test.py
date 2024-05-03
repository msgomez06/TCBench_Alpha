# %% Imports
# OS and IO
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dask.array as da
from dask.ml import preprocessing

# Backend Libraries
import joblib as jl

from utils import toolbox, constants, ML_functions as mlf
from utils import data_lib as dlib

# %%
datadir = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha"

# %% Data Loading
sets, data = toolbox.get_sets(
    {"train": 0.6, "test": 0.2},
    datadir=datadir,
)

# %% Preprocessing
scaling_data = data["train"]["inputs"]

AI_scaler = mlf.AI_StandardScaler()
AI_scaler.fit(scaling_data)

scaler_test = AI_scaler.transform(scaling_data[:100])

# %%
