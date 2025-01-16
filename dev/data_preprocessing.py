# %%
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
            "--overwrite_cache",
            # "--min_leadtime",
            # "6",
            # "--max_leadtime",
            # "24",
            # "--use_gpu",
            # "--verbose",
            # "--reanalysis",
            "--cache_dir",
            "/scratch/mgomezd1/cache",
            "--mask",
            "--magAngle_mode",
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

    args = parser.parse_args()

    print("Imports successful", flush=True)

    # print the multiprocessing start method
    print(torch.multiprocessing.get_start_method(allow_none=True))
    dask.config.set(scheduler="processes")

    #  Setup
    datadir = args.datadir
    cache_dir = args.cache_dir + f"_{args.ai_model}"
    result_dir = args.result_dir

    num_cores = int(subprocess.check_output(["nproc"], text=True).strip())

    #  Data Loading
    rng = np.random.default_rng(seed=2020)

    years = list(range(2013, 2020))
    rng.shuffle(years)

    # Make the cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if args.reanalysis:
        sets, data = toolbox.get_reanal_sets(
            {
                "train": years[:-2],
                "validation": years[-2:],
                # "test": [2020],
            },
            datadir=datadir,
            test_strategy="custom",
            base_position=True,
            cache_dir=cache_dir,
            # use_cached=not args.overwrite_cache,
            verbose=args.verbose,
            debug=args.debug,
        )
    else:
        print("Loading datasets...", flush=True)
        sets, data = toolbox.get_ai_sets(
            {
                "train": years[:-2],
                "validation": years[-2:],
                # "test": [2020],
            },
            datadir=datadir,
            test_strategy="custom",
            base_position=True,
            ai_model=args.ai_model,
            cache_dir=cache_dir,
            # use_cached=not args.overwrite_cache,
            verbose=args.verbose,
            debug=args.debug,
        )

    # %%
    # create a mask for the leadtimes
    train_ldt_mask = da.logical_and(
        data["train"]["leadtime"] >= args.min_leadtime,
        data["train"]["leadtime"] <= args.max_leadtime,
    )
    train_ldt_mask = np.squeeze(train_ldt_mask.compute(scheduler="threads"))

    validation_ldt_mask = da.logical_and(
        data["validation"]["leadtime"] >= args.min_leadtime,
        data["validation"]["leadtime"] <= args.max_leadtime,
    )
    validation_ldt_mask = np.squeeze(validation_ldt_mask.compute(scheduler="threads"))

    # Let's filter the data using leadtimes
    print("Filtering data...", flush=True)
    train_data = data["train"]["inputs"][train_ldt_mask]
    valid_data = data["validation"]["inputs"][validation_ldt_mask]
    # and the columns to ablate
    ablate_cols = json.loads(args.ablate_cols)

    if len(ablate_cols) > 0:
        train_data = train_data[
            :,
            [i for i in range(train_data.shape[1]) if i not in ablate_cols],
            :,
            :,
        ]
        valid_data = valid_data[
            :,
            [i for i in range(valid_data.shape[1]) if i not in ablate_cols],
            :,
            :,
        ]

    if args.magAngle_mode:
        train_data = mlf.uv_to_magAngle(train_data, u_idx=0, v_idx=1)
        valid_data = mlf.uv_to_magAngle(valid_data, u_idx=0, v_idx=1)

    # We will want to normalize the inputs for the model
    # to work properly. We will use the AI_StandardScaler for this
    # purpose
    AI_scaler = None
    from_cache = not args.overwrite_cache
    fpath = os.path.join(cache_dir, "AI_scaler.pkl")

    if from_cache:
        if os.path.exists(fpath):
            print("Loading AI scaler from cache...", flush=True)
            with open(fpath, "rb") as f:
                AI_scaler = pickle.load(f)

    if AI_scaler is None:
        print("Fitting AI datascaler...", flush=True)
        # AI_data = optimize(AI_data)[0]
        AI_scaler = mlf.AI_StandardScaler()
        AI_scaler.fit(train_data, num_workers=num_cores)

        # save the scaler to the cache
        with open(fpath, "wb") as f:
            pickle.dump(AI_scaler, f)

    # We also want to do the same for the base intensity
    base_scaler = None
    from_cache = not args.overwrite_cache
    fpath = os.path.join(cache_dir, "base_scaler.pkl")

    if from_cache:
        if os.path.exists(fpath):
            print("Loading base scaler from cache...", flush=True)
            with open(fpath, "rb") as f:
                base_scaler = pickle.load(f)

    if base_scaler is None:
        print("Fitting base intensity scaler...", flush=True)
        base_scaler = StandardScaler()
        base_scaler.fit(data["train"]["base_intensity"][train_ldt_mask])

        # save the scaler to the cache
        with open(fpath, "wb") as f:
            pickle.dump(base_scaler, f)

    # and one for the base position
    print("Encoding base position...", flush=True)
    train_positions = mlf.latlon_to_sincos(
        data["train"]["base_position"][train_ldt_mask]
    ).compute()
    valid_positions = mlf.latlon_to_sincos(
        data["validation"]["base_position"][validation_ldt_mask]
    ).compute()

    # We'll encode the leadtime by dividing it by the max leadtime in the dataset
    # which is 168 hours
    print("Encoding leadtime...", flush=True)
    max_train_ldt = data["train"]["leadtime"][train_ldt_mask].max().compute()
    train_leadtimes = data["train"]["leadtime"][train_ldt_mask] / max_train_ldt
    validation_leadtimes = (
        data["validation"]["leadtime"][validation_ldt_mask] / max_train_ldt
    )

    if args.raw_target:
        train_target = data["train"]["outputs"][train_ldt_mask]
        valid_target = data["validation"]["outputs"][validation_ldt_mask]
    else:
        # We also want to precalculate the delta intensity for the
        # training and validation sets
        print("Calculating target (i.e., delta intensities)...", flush=True)
        train_target = (
            data["train"]["outputs"][train_ldt_mask]
            - data["train"]["base_intensity"][train_ldt_mask]
        )
        valid_target = (
            data["validation"]["outputs"][validation_ldt_mask]
            - data["validation"]["base_intensity"][validation_ldt_mask]
        )

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
    # Handle the AI and target data in the zarr store
    found = np.all(
        [
            "AIX" in train_arrays,
            "target" in train_arrays,
            "AIX" in valid_arrays,
            "target" in valid_arrays,
        ]
    )

    if not found or args.overwrite_cache:
        dask.config.set(scheduler="threads")
        train_data.rechunk((1000, -1, -1, -1)).to_zarr(
            zarr_store,
            component="train/AIX",
            overwrite=True,
        )
        print("Train data stored in cache...", flush=True)
        train_target.rechunk((1000, -1, -1)).to_zarr(
            zarr_store,
            component="train/target",
            overwrite=True,
        )
        print("Train target stored in cache...", flush=True)
        valid_data.rechunk((1000, -1, -1, -1)).to_zarr(
            zarr_store,
            component="validation/AIX",
            overwrite=True,
        )
        print("Validation data stored in cache...", flush=True)
        valid_target.rechunk(
            (
                1000,
                -1,
                -1,
            )
        ).to_zarr(zarr_store, component="validation/target", overwrite=True)
        print("Validation target stored in cache...", flush=True)
    # %%
    # Load the data from the zarr store
    train_data = da.from_zarr(
        zarr_store, component="train/AIX", chunks=(1000, -1, -1, -1)
    )
    train_target = da.from_zarr(zarr_store, component="train/target")
    valid_data = da.from_zarr(
        zarr_store, component="validation/AIX", chunks=(1000, -1, -1, -1)
    )
    valid_target = da.from_zarr(zarr_store, component="validation/target")

    # %%
    # %%
    # And scale the target data using the cached scaler if available
    target_scaler = None
    from_cache = not args.overwrite_cache
    fpath = os.path.join(cache_dir, "target_scaler.pkl")

    if from_cache:
        if os.path.exists(fpath):
            print("Loading target scaler from cache...", flush=True)
            with open(fpath, "rb") as f:
                target_scaler = pickle.load(f)

    if target_scaler is None:
        print("Fitting target scaler...", flush=True)
        target_scaler = StandardScaler()
        target_scaler.fit(train_target)

        # save the scaler to the cache
        with open(fpath, "wb") as f:
            pickle.dump(target_scaler, f)

    # %% Scaling AI Outputs
    print("Preparing scaled data...", flush=True)
    found = np.all(
        [
            "train/AIX_scaled" in train_arrays,
            "validation/AIX_scaled" in valid_arrays,
            "train/target_scaled" in train_arrays,
            "validation/target_scaled" in valid_arrays,
        ]
    )

    if not found or args.overwrite_cache:
        train_data = da.map_blocks(
            transform_data,
            train_data,
            AI_scaler,
            dtype=train_data.dtype,
            chunks=train_data.chunks,
        )
        valid_data = da.map_blocks(
            transform_data,
            valid_data,
            AI_scaler,
            dtype=valid_data.dtype,
            chunks=valid_data.chunks,
        )
        train_target = da.map_blocks(
            transform_data,
            train_target,
            target_scaler,
            dtype=train_target.dtype,
            chunks=train_target.chunks,
        )
        valid_target = da.map_blocks(
            transform_data,
            valid_target,
            target_scaler,
            dtype=valid_target.dtype,
            chunks=valid_target.chunks,
        )

    train_data.astype(np.float16).to_zarr(
        zarr_store, component="train/AIX_scaled", overwrite=True
    )
    valid_data.astype(np.float16).to_zarr(
        zarr_store, component="validation/AIX_scaled", overwrite=True
    )
    train_target.astype(np.float16).to_zarr(
        zarr_store, component="train/target_scaled", overwrite=True
    )
    valid_target.astype(np.float16).to_zarr(
        zarr_store, component="validation/target_scaled", overwrite=True
    )

    # %%
    # Handle masking in the zarr store if necessary
    found = np.all(
        [
            "masked_AIX" in train_arrays,
            "masked_AIX" in valid_arrays,
            "train/mask" in train_arrays,
            "validation/mask" in valid_arrays,
        ]
    )

    if args.mask and ((not found) or args.overwrite_cache):
        mask_path = args.mask_path
        with open(mask_path, "rb") as f:
            mask_dict = pickle.load(f)[args.mask_type]

        print("Calculating radial mask...", end="", flush=True)
        unique_leads = da.unique(data["train"]["leadtime"]).compute(scheduler="threads")
        train_mask = zr.ones(
            (train_data.shape[0], 1, 241, 241),
            chunks=(1000, 1, 241, 241),
            store=zarr_store,
            path="train/mask",
            overwrite=True,
        )
        valid_mask = zr.ones(
            (valid_data.shape[0], 1, 241, 241),
            chunks=(1000, 1, 241, 241),
            store=zarr_store,
            path="validation/mask",
            overwrite=True,
        )

        for lead in unique_leads:
            radial_mask = mask_dict[lead]

            temp_ldt_mask_train = (
                (data["train"]["leadtime"] == lead).compute().squeeze()
            )
            temp_ldt_mask_valid = (
                (data["validation"]["leadtime"] == lead).compute().squeeze()
            )
            temp_ldt_mask_train = np.nonzero(temp_ldt_mask_train)[0]
            temp_ldt_mask_valid = np.nonzero(temp_ldt_mask_valid)[0]

            train_mask[temp_ldt_mask_train, 0, :, :] = (
                train_mask[temp_ldt_mask_train, 0, :, :] * radial_mask
            )
            valid_mask[temp_ldt_mask_valid, 0, :, :] = (
                valid_mask[temp_ldt_mask_valid, 0, :, :] * radial_mask
            )

        train_mask = da.from_zarr(
            zarr_store, component="train/mask", chunks=(1000, -1, -1, -1)
        )
        valid_mask = da.from_zarr(
            zarr_store, component="validation/mask", chunks=(1000, -1, -1, -1)
        )

        # Function to apply the mask to each block of images
        def apply_mask(image_block, mask_block):
            masked_images = image_block * mask_block
            return masked_images

        masked_train = da.map_blocks(
            apply_mask,
            train_data,
            train_mask,
            dtype=np.float16,  # train_data.dtype,
            chunks=train_data.chunks,
        )
        masked_valid = da.map_blocks(
            apply_mask,
            valid_data,
            valid_mask,
            dtype=np.float16,  # valid_data.dtype,
            chunks=valid_data.chunks,
        )

        masked_train.to_zarr(zarr_store, component="train/masked_AIX", overwrite=True)
        masked_valid.to_zarr(
            zarr_store, component="validation/masked_AIX", overwrite=True
        )

        # Load the data from the zarr store
        train_data = da.from_zarr(
            zarr_store, component="train/masked_AIX", chunks=(1000, -1, -1, -1)
        )
        valid_data = da.from_zarr(
            zarr_store, component="validation/masked_AIX", chunks=(1000, -1, -1, -1)
        )
    elif args.mask and found:
        train_data = da.from_zarr(
            zarr_store, component="train/masked_AIX", chunks=(1000, -1, -1, -1)
        )
        valid_data = da.from_zarr(
            zarr_store, component="validation/masked_AIX", chunks=(1000, -1, -1, -1)
        )
    # %%
    # handle the leadtime in the zarr store
    found = np.all(
        [
            "train/leadtime" in train_arrays,
            "validation/leadtime" in valid_arrays,
        ]
    )

    if not found or args.overwrite_cache:
        data["train"]["leadtime"][train_ldt_mask].rechunk((1000, -1)).to_zarr(
            zarr_store, component="train/leadtime", overwrite=True
        )
        data["validation"]["leadtime"][validation_ldt_mask].rechunk((1000, -1)).to_zarr(
            zarr_store, component="validation/leadtime", overwrite=True
        )
        train_leadtimes.rechunk((1000, -1)).to_zarr(
            zarr_store, component="train/leadtime_scaled", overwrite=True
        )
        validation_leadtimes.rechunk((1000, -1)).to_zarr(
            zarr_store, component="validation/leadtime_scaled", overwrite=True
        )

    # Load the data from the zarr store
    train_leadtimes = da.from_zarr(zarr_store, component="train/leadtime_scaled")
    validation_leadtimes = da.from_zarr(
        zarr_store, component="validation/leadtime_scaled"
    )

    print("Preparing base intensity and position...", flush=True)
    # handle the base intensity and position in the zarr store
    found = np.all(
        [
            "train/base_intensity" in train_arrays,
            "validation/base_intensity" in valid_arrays,
            "train/base_position" in train_arrays,
            "validation/base_position" in valid_arrays,
        ]
    )

    if not found or args.overwrite_cache:
        data["train"]["base_intensity"][train_ldt_mask].rechunk((1000, -1, -1)).to_zarr(
            zarr_store, component="train/base_intensity", overwrite=True
        )
        data["validation"]["base_intensity"][validation_ldt_mask].rechunk(
            (1000, -1, -1)
        ).to_zarr(zarr_store, component="validation/base_intensity", overwrite=True)
        data["train"]["base_position"][train_ldt_mask].rechunk((1000, -1)).to_zarr(
            zarr_store, component="train/base_position", overwrite=True
        )
        data["validation"]["base_position"][validation_ldt_mask].rechunk(
            (1000, -1)
        ).to_zarr(zarr_store, component="validation/base_position", overwrite=True)

    # Load the data from the zarr store
    train_base_intensity = da.from_zarr(zarr_store, component="train/base_intensity")
    valid_base_intensity = da.from_zarr(
        zarr_store, component="validation/base_intensity"
    )
    train_base_position = da.from_zarr(zarr_store, component="train/base_position")
    valid_base_position = da.from_zarr(zarr_store, component="validation/base_position")

    print("Preparing scaled base intensity...", flush=True)
    found = np.all(
        [
            "train/base_intensity_scaled" in train_arrays,
            "validation/base_intensity_scaled" in valid_arrays,
        ]
    )

    if not found or args.overwrite_cache:
        # scale the base intensity using the cached scaler if available
        da.map_blocks(
            transform_data,
            train_base_intensity,
            base_scaler,
            dtype=train_base_intensity.dtype,
            chunks=train_base_intensity.chunks,
        ).to_zarr(zarr_store, component="train/base_intensity_scaled", overwrite=True)
        da.map_blocks(
            transform_data,
            valid_base_intensity,
            base_scaler,
            dtype=valid_base_intensity.dtype,
            chunks=valid_base_intensity.chunks,
        ).to_zarr(
            zarr_store, component="validation/base_intensity_scaled", overwrite=True
        )

    # Load the data from the zarr store
    train_base_intensity = da.from_zarr(
        zarr_store, component="train/base_intensity_scaled"
    )
    valid_base_intensity = da.from_zarr(
        zarr_store, component="validation/base_intensity_scaled"
    )

# %%
