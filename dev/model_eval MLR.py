# %% Important TODO: Separate preprocessing script from training script
# This should be easily doable now with on disk data caching.

# %%
# Imports
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
import json
import zarr as zr

# from dask import optimize
import time


# Backend Libraries
import joblib as jl

from utils import toolbox, ML_functions as mlf
import metrics, baselines

# Importing the sklearn metrics
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import argparse


# Function to Normalize the data and targets
def transform_data(data, scaler):
    return scaler.transform(data)


# %%
if __name__ == "__main__":
    # emulate system arguments
    emulate = True
    # Simulate command line arguments
    if emulate:
        sys.argv = [
            "script_name",  # Traditionally the script name, but it's arbitrary in Jupyter
            # "--ai_model",
            # "fourcastnetv2",
            # "--overwrite_cache",
            # "--min_leadtime",
            # "6",
            # "--max_leadtime",
            # "24",
            # "--use_gpu",
            # "--verbose",
            # "--reanalysis",
            # "--mode",
            # "probabilistic",
            "--cache_dir",
            "/scratch/mgomezd1/cache",
            # "/srv/scratch/mgomezd1/cache",
            "--mask",
            "--magAngle_mode",
            "--dask_array",
        ]
    # %%
    # check if the context has been set for torch multiprocessing
    if torch.multiprocessing.get_start_method() != "spawn":
        torch.multiprocessing.set_start_method("spawn", force=True)

    # Read in arguments with argparse
    parser = argparse.ArgumentParser(description="Train an MLR model")
    parser.add_argument(
        "--datadir",
        type=str,
        default="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha",
        help="Directory where the data is stored",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/scratch/mgomezd1/cache",
        help="Directory where the cache is stored",
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        default="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/",
        help="Directory where the results are stored",
    )

    parser.add_argument(
        "--ignore_gpu",
        action="store_true",
        help="Whether to ignore the GPU for training",
    )

    parser.add_argument(
        "--ai_model",
        type=str,
        default="panguweather",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="deterministic",
    )

    parser.add_argument(
        "--deterministic_loss",
        type=str,
        default="MSE",
    )

    parser.add_argument(
        "--overwrite_cache",
        help="Enable cache overwriting",
        action="store_true",
    )

    parser.add_argument(
        "--verbose",
        help="Enable verbose mode",
        action="store_true",
    )

    parser.add_argument(
        "--debug",
        help="Enable debug mode",
        action="store_true",
    )

    parser.add_argument(
        "--min_leadtime",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--max_leadtime",
        type=int,
        default=168,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--ablate_cols",
        type=str,
        default="[]",
    )

    parser.add_argument(
        "--fc_width",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--magAngle_mode",
        action="store_true",
        help="Whether to use magnitude and angle instead of u and v",
    )

    parser.add_argument(
        "--raw_target",
        action="store_true",
        help="Whether to use intensity instead of intensification",
    )

    parser.add_argument(
        "--reanalysis",
        action="store_true",
        help="Whether to use reanalysis data instead of forecast data",
    )

    parser.add_argument(
        "--mask",
        action="store_true",
        help="Whether to apply a leadtime driven mask to the data",
    )

    parser.add_argument(
        "--mask_path",
        type=str,
        default="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/mask_dict.pkl",
    )

    parser.add_argument(
        "--mask_type",
        type=str,
        default="linear",
    )

    parser.add_argument(
        "--aux_loss",
        action="store_true",
        help="Whether to use an auxiliary loss",
    )

    parser.add_argument(
        "--dask_array",
        action="store_true",
        help="Whether to use dask arrays for the dataset",
    )

    args = parser.parse_args()

    print("Imports successful", flush=True)

    # print the multiprocessing start method
    print(torch.multiprocessing.get_start_method(allow_none=True))
    dask.config.set(scheduler="processes")

    #  Setup
    datadir = args.datadir
    cache_dir = args.cache_dir + f"_{args.ai_model}"
    result_dir = args.result_dir

    # Check for GPU availability
    if torch.cuda.is_available() and not args.ignore_gpu:
        calc_device = torch.device("cuda:0")
    else:
        calc_device = torch.device("cpu")

    num_cores = int(subprocess.check_output(["nproc"], text=True).strip())

    # %%
    # Check if the cache directory includes a zarray store for the data
    zarr_path = os.path.join(cache_dir, "zarray_store")
    if not os.path.exists(zarr_path):
        os.makedirs(zarr_path)
    zarr_store = zr.DirectoryStore(zarr_path)

    # Check if the root group exists, if not create it
    root = zr.group(zarr_store)
    root = zr.open_group(zarr_path)

    # Check that the training and validation groups exist, if not create them
    if "train" not in root:
        root.create_group("train")
    if "validation" not in root:
        root.create_group("validation")

    # Check if the data is already stored in the cache
    train_zarr = root["train"]
    valid_zarr = root["validation"]

    train_arrays = list(train_zarr.array_keys())
    valid_arrays = list(valid_zarr.array_keys())
    # %%

    # assert that AIX_masked, target_scaled, base_intensity_scaled, base_position, leadtime_scaled are in each group
    assert np.all(
        [
            # "masked_AIX" in valid_arrays,
            "target_scaled" in valid_arrays,
            "base_intensity_scaled" in valid_arrays,
            "base_position" in valid_arrays,
            "leadtime_scaled" in valid_arrays,
            "leadtime" in valid_arrays,
        ]
    ), "Missing Data in the cache"

    if args.dask_array:
        # Load the data from the zarr store using dask array
        valid_data = da.from_zarr(zarr_store, component="validation/masked_AIX")
        unmasked_valid_data = da.from_zarr(
            zarr_store, component="validation/AIX_scaled"
        )
        valid_target = da.from_zarr(zarr_store, component="validation/target_scaled")
        validation_leadtimes = da.from_zarr(
            zarr_store, component="validation/leadtime_scaled"
        )

        valid_base_intensity = da.from_zarr(
            zarr_store, component="validation/base_intensity_scaled"
        )
        valid_base_position = da.from_zarr(
            zarr_store, component="validation/base_position"
        )
        validation_unscaled_leadtimes = da.from_zarr(
            zarr_store, component="validation/leadtime"
        )
        # Load the training targets and unscaled leadtimes for climatology
        train_target = da.from_zarr(zarr_store, component="train/target_scaled")
        train_leadtimes = da.from_zarr(zarr_store, component="train/leadtime")
    else:
        # load data with zarr
        valid_data = valid_zarr["masked_AIX"]
        unmasked_valid_data = valid_zarr["AIX_scaled"]
        valid_target = valid_zarr["target_scaled"]
        validation_leadtimes = valid_zarr["leadtime_scaled"]
        valid_base_intensity = valid_zarr["base_intensity_scaled"]
        valid_base_position = valid_zarr["base_position"]
        validation_unscaled_leadtimes = valid_zarr["leadtime"]
        # Load the training targets and unscaled leadtimes for climatology
        train_target = train_zarr["target_scaled"]
        train_leadtimes = train_zarr["leadtime"]

    # %%
    valid_maxima = valid_data.max(axis=(-2, -1)).compute(scheduler="threads")
    valid_minima = valid_data.min(axis=(-2, -1)).compute(scheduler="threads")
    unmasked_valid_maxima = unmasked_valid_data.max(axis=(-2, -1)).compute(
        scheduler="threads"
    )
    unmasked_valid_minima = unmasked_valid_data.min(axis=(-2, -1)).compute(
        scheduler="threads"
    )

    valid_range = valid_maxima - valid_minima
    unmasked_valid_range = unmasked_valid_maxima - unmasked_valid_minima

    # %%
    validation_leadtimes = validation_leadtimes.compute(scheduler="threads")
    valid_base_intensity = valid_base_intensity.compute(scheduler="threads")
    validation_unscaled_leadtimes = validation_unscaled_leadtimes.compute(
        scheduler="threads"
    )
    unique_leadtimes = np.unique(validation_unscaled_leadtimes)
    valid_target = valid_target.compute(scheduler="threads")

    train_target = train_target.compute(scheduler="threads")
    train_leadtimes = train_leadtimes.compute(scheduler="threads")

    # var order = ["W_mag", "W_dir", "mslp", "Z500", "T850"]

    # %%
    valid_x = np.vstack(
        [
            valid_maxima[:, 0],  # Maximum wind magnitude
            valid_minima[:, 2],  # Minimum mean sea level pressure
            valid_range[:, 0],  # Range of wind magnitude
            valid_range[:, 2],  # Range of mean sea level pressure
            valid_minima[:, 3],  # Minimum geopotential height at 500 hPa
            valid_range[:, 4],  # Range of temperature at 850 hPa
            validation_leadtimes.squeeze(),  # Leadtime
            valid_base_intensity.T,  # Base intensity
        ]
    ).T

    unmasked_valid_x = np.vstack(
        [
            unmasked_valid_maxima[:, 0],  # Maximum wind magnitude
            unmasked_valid_minima[:, 2],  # Minimum mean sea level pressure
            unmasked_valid_range[:, 0],  # Range of wind magnitude
            unmasked_valid_range[:, 2],  # Range of mean sea level pressure
            unmasked_valid_minima[:, 3],  # Minimum geopotential height at 500 hPa
            unmasked_valid_range[:, 4],  # Range of temperature at 850 hPa
            validation_leadtimes.squeeze(),  # Leadtime
            valid_base_intensity.T,  # Base intensity
        ]
    ).T

    # %%
    # Defining the models to load and the hyperparameters needed to evaluate them
    results_dir = (
        "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/"
    )
    models = [
        {
            "filepath": f"{results_dir}best_model_TorchMLR_12-12-15h15_epoch-40_panguweather_deterministic MagUnmasked.pt",
            "masked": False,
            "probabilistic": False,
            "tag": "unmasked MLR",
            "results": [],
        },
        {
            "filepath": f"{results_dir}best_model_TorchMLR_12-12-15h00_epoch-40_panguweather_probabilistic MagUnmasked.pt",
            "masked": False,
            "probabilistic": True,
            "tag": "unmasked MLR",
            "results": [],
        },
        {
            "filepath": f"{results_dir}best_model_TorchMLR_12-12-14h47_epoch-40_panguweather_probabilistic MagMasked.pt",
            "masked": True,
            "probabilistic": True,
            "tag": "masked MLR",
            "results": [],
        },
        {
            "filepath": f"{results_dir}best_model_TorchMLR_12-12-14h37_epoch-40_panguweather_deterministic MagMasked.pt",
            "masked": True,
            "probabilistic": False,
            "tag": "masked MLR",
            "results": [],
        },
    ]

    # Load each pytorch model into the models dictionary
    for model in models:
        model["model"] = torch.load(
            os.path.join(results_dir, model["filepath"]), map_location=calc_device
        )
    # %%
    # Train the climatology model
    climatology_model = baselines.AveClimatology()
    climatology_model.fit(
        target=train_target,
        leadtimes=train_leadtimes.squeeze(),
    )

    # %%
    # Evaluate the models
    climatology_results = {"deterministic": [], "probabilistic": []}
    persistence_results = {"deterministic": [], "probabilistic": []}
    for unique_leadtime in unique_leadtimes:
        bool_idxs = (validation_unscaled_leadtimes == unique_leadtime).squeeze()
        temp_x = valid_x[bool_idxs]
        temp_unmasked_x = unmasked_valid_x[bool_idxs]
        temp_target = valid_target[bool_idxs]

        temp_clim = climatology_model.predict([unique_leadtime]).flatten()

        clim_prob = np.tile(temp_clim, (temp_target.shape[0], 1))
        clim_det = np.tile(temp_clim[::2], (temp_target.shape[0], 1))

        temp_persistence = np.zeros_like(temp_target)

        deterministic_loss = (
            torch.nn.MSELoss()
            if args.deterministic_loss == "MSE"
            else torch.nn.L1Loss()
        )

        deterministic_score = deterministic_loss(
            torch.tensor(clim_det, dtype=torch.float32),
            torch.tensor(temp_target, dtype=torch.float32),
        ).item()
        persistence_score = deterministic_loss(
            torch.tensor(temp_persistence, dtype=torch.float32),
            torch.tensor(temp_target, dtype=torch.float32),
        ).item()
        mae_persistence = torch.nn.L1Loss()(
            torch.tensor(temp_persistence, dtype=torch.float32),
            torch.tensor(temp_target, dtype=torch.float32),
        ).item()

        climatology_results["deterministic"].append(
            deterministic_loss(
                torch.tensor(clim_det, dtype=torch.float32),
                torch.tensor(temp_target, dtype=torch.float32),
            ).item()
        )
        climatology_results["probabilistic"].append(
            metrics.CRPS_ML(
                torch.tensor(clim_prob, dtype=torch.float32),
                torch.tensor(temp_target, dtype=torch.float32),
            ).item()
        )
        persistence_results["deterministic"].append(persistence_score)
        persistence_results["probabilistic"].append(mae_persistence)

        for model in models:
            if model["probabilistic"]:
                loss_func = metrics.CRPS_ML
            else:
                if args.deterministic_loss == "MSE":
                    loss_func = torch.nn.MSELoss()
                elif args.deterministic_loss == "MAE":
                    loss_func = torch.nn.L1Loss()
                else:
                    raise ValueError("Loss function not recognized.")
            with torch.no_grad():
                if model["masked"]:
                    prediction = model["model"](
                        torch.tensor(temp_x, dtype=torch.float32)
                    )
                else:
                    prediction = model["model"](
                        torch.tensor(temp_unmasked_x, dtype=torch.float32)
                    )
                loss = loss_func(
                    prediction, torch.tensor(temp_target, dtype=torch.float32)
                )
                model["results"].append(loss.item())
                loss = None

# %% Plotting the results

fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=150)


colors = (
    np.array([38, 1, 36]) / 255,  # Black
    np.array([242, 141, 168]) / 255,  # Pink
    np.array([7, 140, 91]) / 255,  # Green
    np.array([242, 179, 102]) / 255,  # Orange
    np.array([242, 241, 240]) / 255,  # White
)
axs[0].set_prop_cycle(color=colors)
axs[1].set_prop_cycle(color=colors)

for model in models:
    if model["probabilistic"]:
        axs[0].plot(
            unique_leadtimes,
            model["results"],
            label=model["tag"],
            marker="o",
            linestyle="-",
        )
    else:
        axs[1].plot(
            unique_leadtimes,
            model["results"],
            label=model["tag"],
            marker="o",
            linestyle="-",
        )

axs[0].plot(
    unique_leadtimes,
    climatology_results["probabilistic"],
    label="Average Climatology",
    marker="o",
    linestyle="-",
)
axs[0].plot(
    unique_leadtimes,
    persistence_results["probabilistic"],
    label="Persistence (MAE)",
    marker="x",
    linestyle="--",
)
axs[1].plot(
    unique_leadtimes,
    climatology_results["deterministic"],
    label="Average Climatology",
    marker="o",
    linestyle="-",
)
axs[1].plot(
    unique_leadtimes,
    persistence_results["deterministic"],
    label="Persistence",
    marker="x",
    linestyle="--",
)


axs[0].set_title("Probabilistic Models")
axs[1].set_title("Deterministic Models")
axs[0].set_xlabel("Leadtime (hours)")
axs[1].set_xlabel("Leadtime (hours)")
axs[0].set_ylabel("CRPS")
axs[1].set_ylabel("MSE Loss")
axs[0].legend()
axs[1].legend()

toolbox.plot_facecolors(fig=fig, axes=axs)

# %%
