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


# %%

# ERA_MLR = '/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/best_model_TorchMLR_09-03-15h52_epoch-5_ERA5_probabilistic.pt'
# PANGU_MLR = '/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/best_model_TorchMLR_08-08-16h24_epoch-11_panguweather_probabilistic.pt'

# pangu = torch.load(PANGU_MLR, map_location=torch.device('cpu'))
# era = torch.load(ERA_MLR, map_location=torch.device('cpu'))


# variables = ["u_max", "u_min", "v_max", "v_min", "mslp_max", "mslp_min", "z500_min", "z500_max", "t850_max", "t850_min", "base_V", "base_minSLP", "lat_sin", "lat_cos", "lon_sin", "lon_cos", "leadtime"]

# era_weights = era.linear.weight.detach().numpy()
# pangu_weights = pangu.linear.weight.detach().numpy()

# def format_weights(weights, variables):
#     return " ".join([f"{'+' if w >= 0 else '-'} {abs(w):.2f} * {v}" for w, v in zip(weights, variables)])

# print("Wind Intensification Rate (mu)= " + format_weights(era_weights[0], variables))
# print("Wind Intensification Rate (sigma)= " + format_weights(era_weights[1], variables))
# print("Pressure Intensification Rate (mu)= " + format_weights(era_weights[2], variables))
# print("Pressure Intensification Rate (Sigma)= " + format_weights(era_weights[3], variables))

# print("Wind Intensification Rate (mu)= " + format_weights(pangu_weights[0], variables))
# print("Wind Intensification Rate (sigma)= " + format_weights(pangu_weights[1], variables))
# print("Pressure Intensification Rate (mu)= " + format_weights(pangu_weights[2], variables))
# print("Pressure Intensification Rate (Sigma)= " + format_weights(pangu_weights[3], variables))

# print("Delta Wind Intensification Rate (mu)= " + format_weights(era_weights[0] - pangu_weights[0], variables))
# print("Delta Wind Intensification Rate (sigma)= " + format_weights(era_weights[1] - pangu_weights[1], variables))
# print("Delta Pressure Intensification Rate (mu)= " + format_weights(era_weights[2] - pangu_weights[2], variables))
# print("Delta Pressure Intensification Rate (Sigma)= " + format_weights(era_weights[3] - pangu_weights[3], variables))


# %%


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
            "--cache_dir",
            "/scratch/mgomezd1/cache",
            # "/srv/scratch/mgomezd1/cache",
            "--mask",
            "--magAngle_mode",
            "--saved_model",
            "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/best_model_Regularized_CNN_11-14-13h43_epoch-14_panguweather_deterministic_[32,64,128].pt",
        ]
    # %%
    # check if the context has been set for torch multiprocessing
    if torch.multiprocessing.get_start_method() != "spawn":
        torch.multiprocessing.set_start_method("spawn", force=True)

    # Read in arguments with argparse
    parser = argparse.ArgumentParser(description="Train a CNN model")
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
        default="RMSE",
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
        "--cnn_width",
        type=str,
        default="[32,64,128]",
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

    parser.add_argument(
        "--saved_model",
        type=str,
        default=None,
        help="Path to the saved model to load",
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

    # assert that AIX_masked, target_scaled, base_intensity_scaled, base_position, leadtime_scaled are in each group
    assert np.all(
        [
            "masked_AIX" in train_arrays,
            "target_scaled" in train_arrays,
            "base_intensity_scaled" in train_arrays,
            "base_position" in train_arrays,
            "leadtime_scaled" in train_arrays,
            "leadtime" in train_arrays,
            "masked_AIX" in valid_arrays,
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
        valid_target = da.from_zarr(zarr_store, component="validation/target_scaled")
        train_leadtimes = da.from_zarr(zarr_store, component="train/leadtime_scaled")
        validation_leadtimes = da.from_zarr(
            zarr_store, component="validation/leadtime_scaled"
        )
        train_base_intensity = da.from_zarr(
            zarr_store, component="train/base_intensity_scaled"
        )
        valid_base_intensity = da.from_zarr(
            zarr_store, component="validation/base_intensity_scaled"
        )
        train_base_position = da.from_zarr(zarr_store, component="train/base_position")
        valid_base_position = da.from_zarr(
            zarr_store, component="validation/base_position"
        )
        train_unscaled_leadtimes = da.from_zarr(zarr_store, component="train/leadtime")
        validation_unscaled_leadtimes = da.from_zarr(
            zarr_store, component="validation/leadtime"
        )
    else:
        # load data with zarr
        valid_data = valid_zarr["masked_AIX"]
        valid_target = valid_zarr["target_scaled"]
        validation_leadtimes = valid_zarr["leadtime_scaled"]
        valid_base_intensity = valid_zarr["base_intensity_scaled"]
        valid_base_position = valid_zarr["base_position"]
        validation_unscaled_leadtimes = valid_zarr["leadtime"]

    # %%
    #  Dataloader & Hyperparameters

    # Let's define some hyperparameters
    batch_size = 256

    # If the mode is not deterministic, we'll set the loss to CRPS
    if args.mode != "deterministic":
        loss_func = metrics.CRPS_ML
    else:
        if args.deterministic_loss == "RMSE":
            loss_func = torch.nn.MSELoss()
        elif args.deterministic_loss == "MAE":
            loss_func = torch.nn.L1Loss()
        else:
            raise ValueError("Loss function not recognized.")

    num_workers = (
        int(num_cores * 2 / 3)
        if (calc_device == torch.device("cpu") and num_cores > 1)
        else num_cores
    )

    #  Model
    # Load the saved model
    model_path = os.path.join(result_dir, args.saved_model)
    ml_model = torch.load(model_path, map_location=calc_device)

    # Store the unique leadtimes
    unique_leadtimes = np.unique(validation_unscaled_leadtimes)
    # %%
    val_losses = []
    if args.aux_loss:
        aux_losses = []
    for unique_leadtime in unique_leadtimes:
        indices = np.isin(validation_unscaled_leadtimes, unique_leadtime).squeeze()
        indices = np.nonzero(indices)[0]

        temp_data = valid_data[indices]
        temp_target = valid_target[indices]
        temp_leadtime = validation_leadtimes[indices]
        temp_base_intensity = valid_base_intensity[indices]
        temp_base_position = valid_base_position[indices]

        # Instantiate the zarrv2 dataset
        temp_dataset = mlf.ZarrDatasetv2(
            AI_X=temp_data,
            base_int=temp_base_intensity,
            target_data=temp_target,
            device=calc_device,
            num_workers=1,
            track=temp_base_position,
            leadtimes=temp_leadtime,
        )

        # Instantiate the dataloader
        temp_loader = mlf.make_dataloader(
            temp_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
        )

        # # Evaluate the model
        # with torch.no_grad():
        #     for AIX, scalars, target in temp_loader:
        #         AIX = AIX.to(calc_device)
        #         scalars = scalars.to(calc_device)
        #         target = target.to(calc_device)

        #         prediction = ml_model(x=AIX, scalars=scalars)

        #         if args.mode != "deterministic":
        #             loss = loss_func(prediction, target)
        #         else:
        #             loss = loss_func(prediction, target)

        #         if args.aux_loss:
        #             val_losses.append(
        #                 loss_func(
        #                     prediction[:, : ml_model.target_size],
        #                     target,
        #                 )
        #             )
        #             aux_losses.append(
        #                 loss_func(
        #                     prediction[:, ml_model.target_size :],
        #                     val_losses[-1],
        #                 )
        #             )
        #         else:
        #             val_losses.append(loss)

        val_loss = 0
        with torch.no_grad():
            i = 0
            for AI_X, scalars, target in temp_loader:
                AI_X = AI_X.to(calc_device)
                scalars = scalars.to(calc_device)
                target = target.to(calc_device)
                AI_X = AI_X.float()
                scalars = scalars.float()
                target = target.float()

                i += 1
                prediction = ml_model(x=AI_X, scalars=scalars)
                if args.aux_loss:
                    batch_loss = loss_func(
                        prediction[:, : ml_model.target_size], target
                    )
                    aux_loss = loss_func(
                        prediction[:, ml_model.target_size :], batch_loss
                    )
                    val_loss += batch_loss.item() + aux_loss.item()
                else:
                    batch_loss = loss_func(prediction, target)

                val_loss += batch_loss.item()

                # print a simple progress bar that shows a dot for each percent of the batch
                print(
                    "\r"
                    + f"Batch val loss: {(val_loss/(i+1)):.4f}, "
                    + f"{i}/{len(temp_loader)}"
                    + "." * int(20 * i / len(temp_loader)),
                    end="",
                    flush=True,
                )

        val_loss = val_loss / len(temp_loader)
        val_losses.append(val_loss)

    # %%
    # Save the results
    results_path = os.path.join(result_dir, "evaluation_results.pkl")

    #  Save the results
    print("Saving results...", flush=True)
    results = {
        "lead_times": unique_leadtimes,
        "val_losses": val_losses,
    }
    with open(os.path.join(result_dir, "leadtime_decomposition.pkl"), "wb") as f:
        pickle.dump(results, f)
    # %%
    #  Plotting
    print("Plotting...", flush=True)
    fig, ax = plt.subplots()
    ax.plot(unique_leadtimes, val_losses)
    ax.set_xlabel("Leadtime (h)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss vs. Leadtime")
    fig.savefig(os.path.join(result_dir, "eval_results.png"))
    # %%
