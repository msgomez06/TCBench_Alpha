# %% Imports
# OS and IO
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

# Backend Libraries
import joblib as jl

from utils import toolbox, constants
from utils import data_lib as dlib

# %% Load the seasons to process

seasons = toolbox.get_TC_seasons(
    season_list=[2020],
    filepath="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha",
)

# %%
# Load the data collection
data_dir = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/AI-milton/panguweather"
dc = dlib.AI_Data_Collection(data_dir)

# %%
# Process the tracks
for season, storms in seasons.items():
    print(f"Starting to process {season}. which contains {len(storms)} storms...")

    for storm in storms:
        print(
            f"Storm: {storm.uid} exists: {os.path.exists(os.path.join(storm.filepath, storm.uid + '.AI.panguweather.nc'))}"
        )

    # # determine the number of processors that can be used
    # n_jobs = jl.cpu_count()

    # # process the tracks
    # jl.Parallel(n_jobs=n_jobs)(
    #     jl.delayed(storm.process_data_collection)(dc) for storm in storms
    # )


# %%
