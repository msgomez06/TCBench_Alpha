# %% Imports and preliminaries
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.toolbox import *

# Define colorblind safe colors for plotting
colors = [
    np.array([215, 166, 122]) / 255,
    np.array([0, 148, 199]) / 255,
    np.array([214, 7, 114]) / 255,
    np.array([50, 50, 50]) / 255,
]

# %% Load deterministic results
results_dir = (
    "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/"
)


pangu_mlrpath = os.path.join(results_dir, "panguweather_deterministic_eval_results.pkl")
era_mlrpath = os.path.join(results_dir, "ERA5_deterministic_eval_results.pkl")
persistence_path = os.path.join(results_dir, "persistence_eval_results.pkl")
clim_path = os.path.join(results_dir, "clim_eval_results.pkl")

with open(pangu_mlrpath, "rb") as f:
    pangu_mlr = pickle.load(f)

with open(era_mlrpath, "rb") as f:
    era_mlr = pickle.load(f)

with open(persistence_path, "rb") as f:
    persistence = pickle.load(f)

with open(clim_path, "rb") as f:
    clim = pickle.load(f)

# %% Load probabilistic results
pangu_probpath = os.path.join(
    results_dir, "panguweather_probabilistic_eval_results.pkl"
)
era_probpath = os.path.join(results_dir, "ERA5_probabilistic_eval_results.pkl")

with open(pangu_probpath, "rb") as f:
    pangu_prob = pickle.load(f)

with open(era_probpath, "rb") as f:
    era_prob = pickle.load(f)

# %% Plot deterministic results

fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

# Plot deterministic results
ax[0].plot(
    pangu_mlr["lead_times"],
    pangu_mlr["val_losses"],
    label="PanguWeather MLR",
    color=colors[0],
)
ax[0].plot(
    era_mlr["lead_times"], era_mlr["val_losses"], label="ERA5 MLR", color=colors[1]
)
ax[0].plot(
    persistence["lead_times"],
    persistence["val_losses"],
    label="Persistence",
    color=colors[2],
)
ax[0].plot(
    clim["lead_times"],
    clim["val_losses"],
    label="Ave. Intensification in Train set",
    color=colors[3],
    linestyle="--",
)

ax[0].set_title("Deterministic Forecast Skill")
ax[0].set_xlabel("Lead Time (hours)")
ax[0].set_ylabel("RMSE (z-score)")
ax[0].legend()

# Plot probabilistic results
ax[1].plot(
    pangu_prob["lead_times"],
    pangu_prob["val_losses"],
    label="PanguWeather Probabilistic MLR",
    color=colors[0],
)
ax[1].plot(
    era_prob["lead_times"],
    era_prob["val_losses"],
    label="ERA5 Probabilistic MLR",
    color=colors[1],
)
ax[1].set_title("Probabilistic Forecast Skill")
ax[1].set_xlabel("Lead Time (hours)")
ax[1].set_ylabel("CRPS")
ax[1].legend()

plot_facecolors(fig=fig, axes=ax)

# %%
