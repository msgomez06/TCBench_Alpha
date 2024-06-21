# %% Imports
# OS and IO
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dask.array as da

# Backend Libraries
import joblib as jl

from utils import toolbox, constants
from utils import data_lib as dlib

# %% Load the seasons to process

seasons = toolbox.get_TC_seasons(
    season_list=[*range(2016, 2021)],
    datadir_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha",
)

# %% Control flags

process = True
perfect_prog = True

# %% Process the tracks
input_samples = None
target_samples = None
for season, storms in seasons.items():
    print(f"Starting to process {season}. which contains {len(storms)} storms...")

    if process:
        # Load the data collection
        data_dir = (
            "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/AI-milton/panguweather"
        )
        dc = dlib.AI_Data_Collection(data_dir)

        # determine the number of processors that can be used
        n_jobs = jl.cpu_count()

        data_dir = (
            "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/AI-milton/panguweather"
        )
        dc = dlib.AI_Data_Collection(data_dir)

        # process the tracks
        jl.Parallel(n_jobs=n_jobs)(
            jl.delayed(storm.process_data_collection)(dc) for storm in storms
        )

    else:
        # for storm in storms:
        #     inputs = None
        #     outputs = None
        #     try:
        #         inputs, outputs, t, leads = storm.serve_ai_data()
        #     except Exception as e:
        #         print(f"Failed to process {str(storm)}")
        #         print(f"Error: {e}")
        #     if inputs is not None and outputs is not None:
        #         if input_samples is None:
        #             input_samples = inputs
        #             target_samples = outputs
        #         else:
        #             input_samples = da.vstack((input_samples, inputs))
        #             target_samples = da.vstack((target_samples, outputs))
        pass

# %%
