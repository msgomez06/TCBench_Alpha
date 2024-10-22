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
import multiprocessing
import json

# from dask import optimize
import time


# Backend Libraries
import joblib as jl

from utils import toolbox, ML_functions as mlf
import metrics, baselines

# Importing the sklearn metrics
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import argparse

if __name__ == "__main__":
    # emulate system arguments
    emulate = False
    # Simulate command line arguments
    if emulate:
        sys.argv = [
            "script_name",  # Traditionally the script name, but it's arbitrary in Jupyter
            "--ai_model",
            "fourcastnetv2",
            "--overwrite_cache",
            # "--min_leadtime",
            # "6",
            # "--max_leadtime",
            # "24",
            "--use_gpu",
            "--verbose",
        ]

    # Read in arguments with argparse
    parser = argparse.ArgumentParser(description="Evaluate an MLR model")
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
        "--saved_model",
        type=str,
        default=None,
        help="Path to the saved model to load",
    )

    parser.add_argument(
        "--aux_loss",
        action="store_true",
        help="Whether an auxiliary loss was used during training",
    )

    args = parser.parse_args()

    print("Imports successful", flush=True)

    # check if the context has been set for torch multiprocessing
    if not torch.multiprocessing.get_start_method(allow_none=True) == "spawn":
        torch.multiprocessing.set_start_method("spawn")

    #  Setup
    datadir = args.datadir
    cache_dir = args.cache_dir + f"_{args.ai_model}"
    result_dir = args.result_dir
    dask.config.set(scheduler="threads")

    # Check for GPU availability
    if torch.cuda.is_available() and not args.ignore_gpu:
        calc_device = torch.device("cuda:0")
    else:
        calc_device = torch.device("cpu")

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

    # create a mask for the leadtimes
    train_ldt_mask = da.logical_and(
        (data["train"]["leadtime"] >= args.min_leadtime),
        (data["train"]["leadtime"] <= args.max_leadtime),
    )
    train_ldt_mask = np.squeeze(train_ldt_mask.compute())

    validation_ldt_mask = da.logical_and(
        (data["validation"]["leadtime"] >= args.min_leadtime),
        (data["validation"]["leadtime"] <= args.max_leadtime),
    )
    validation_ldt_mask = np.squeeze(validation_ldt_mask.compute())

    #  Preprocessing

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
    train_leadtimes = (
        data["train"]["leadtime"][train_ldt_mask] / max_train_ldt
    ).compute()
    validation_leadtimes = (
        data["validation"]["leadtime"][validation_ldt_mask] / max_train_ldt
    ).compute()

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

    #  Dataloader & Hyperparameters

    # Let's define some hyperparameters
    batch_size = 32

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
    dask.config.set(scheduler="threads")

    unique_leads = da.unique(data["train"]["leadtime"]).compute()

    if emulate:
        input("Press Enter to continue...")

    #  Model
    # Load the saved model
    model_path = os.path.join(result_dir, args.saved_model)
    MLR = torch.load(model_path)

    #  Training
    lead_times = []
    val_losses = []

    for lead in unique_leads:
        print(f"Working on Leadtime: {lead}h")

        lead_times.append(lead)

        temp_mask = data["validation"]["leadtime"] == lead
        temp_mask = np.squeeze(temp_mask.compute())

        # We then instantiate the DaskDataset class for the training and validation sets.
        # Validation first because it's smaller and will be used to evaluate if the code
        # is working as expected
        print("Creating validation DaskDataset and dataloader...", flush=True)
        validation_dataset = mlf.DaskDataset(
            AI_X=valid_data[temp_mask],
            AI_scaler=AI_scaler,
            base_int=data["validation"]["base_intensity"][validation_ldt_mask][
                temp_mask
            ],
            base_scaler=base_scaler,
            target_data=valid_target[temp_mask],
            target_scaler=target_scaler,
            device=calc_device,
            cachedir=cache_dir,
            zarr_name=f"val_{lead}h",
            overwrite=args.overwrite_cache,
            num_workers=num_cores,
            track=valid_positions[temp_mask],
            leadtimes=validation_leadtimes[temp_mask],
            chunk_size=512,
            # load_into_memory=True,
        )

        validation_loader = mlf.make_dataloader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        val_loss = 0
        with torch.no_grad():
            i = 0
            for AI_X, scalars, target in validation_loader:
                i += 1
                prediction = MLR(x=AI_X, scalars=scalars)
                batch_loss = loss_func(prediction, target)

                val_loss += batch_loss.item()

                # print a simple progress bar that shows a dot for each percent of the batch
                print(
                    "\r"
                    + f"Batch val loss: {val_loss:.4f}, "
                    + f"{i}/{len(validation_loader)}"
                    + "." * int(20 * i / len(validation_loader)),
                    end="",
                    flush=True,
                )

        val_loss = val_loss / len(validation_loader)
        val_losses.append(val_loss)

        print(
            f"\nLead: {lead},",
            f"val_loss: {val_loss},",
            flush=True,
            sep=" ",
        )

    #  Save the results
    print("Saving results...", flush=True)
    results = {
        "lead_times": lead_times,
        "val_losses": val_losses,
    }

    with open(os.path.join(result_dir, "eval_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    print("Done!", flush=True)

    #  Plotting
    print("Plotting...", flush=True)
    fig, ax = plt.subplots()
    ax.plot(lead_times, val_losses)
    ax.set_xlabel("Leadtime (h)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss vs. Leadtime")
    fig.savefig(os.path.join(result_dir, "eval_results.png"))
