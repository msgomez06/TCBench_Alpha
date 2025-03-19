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
    emulate = False
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
            "--algorithm",
            "UNet",
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
        "--algorithm",
        type=str,
        default="CNN",
    )

    parser.add_argument(
        "--l1_reg",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="",
    )

    args = parser.parse_args()

    print("Imports successful", flush=True)
    # %%
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

    # if args.dask_array:
    #     # Load the data from the zarr store using dask array
    #     if args.mask:
    #         train_data = da.from_zarr(zarr_store, component="train/masked_AIX")
    #         valid_data = da.from_zarr(zarr_store, component="validation/masked_AIX")
    #     else:
    #         train_data = da.from_zarr(zarr_store, component="train/AIX_scaled")
    #         valid_data = da.from_zarr(zarr_store, component="validation/AIX_scaled")
    #     train_target = da.from_zarr(zarr_store, component="train/target_scaled")

    #     valid_target = da.from_zarr(zarr_store, component="validation/target_scaled")
    #     train_leadtimes = da.from_zarr(zarr_store, component="train/leadtime_scaled")
    #     validation_leadtimes = da.from_zarr(
    #         zarr_store, component="validation/leadtime_scaled"
    #     )
    #     train_base_intensity = da.from_zarr(
    #         zarr_store, component="train/base_intensity_scaled"
    #     )
    #     valid_base_intensity = da.from_zarr(
    #         zarr_store, component="validation/base_intensity_scaled"
    #     )
    #     train_base_position = da.from_zarr(zarr_store, component="train/base_position")
    #     valid_base_position = da.from_zarr(
    #         zarr_store, component="validation/base_position"
    #     )
    #     train_unscaled_leadtimes = da.from_zarr(zarr_store, component="train/leadtime")
    #     validation_unscaled_leadtimes = da.from_zarr(
    #         zarr_store, component="validation/leadtime"
    #     )
    # else:
    #     # load data with zarr
    #     if args.mask:
    #         train_data = train_zarr["masked_AIX"]
    #         valid_data = valid_zarr["masked_AIX"]
    #     else:
    #         train_data = train_zarr["AIX_scaled"]
    #         valid_data = valid_zarr["AIX_scaled"]
    #     train_target = train_zarr["target_scaled"]
    #     valid_target = valid_zarr["target_scaled"]
    #     train_leadtimes = train_zarr["leadtime_scaled"]
    #     validation_leadtimes = valid_zarr["leadtime_scaled"]
    #     train_base_intensity = train_zarr["base_intensity_scaled"]
    #     valid_base_intensity = valid_zarr["base_intensity_scaled"]
    #     train_base_position = train_zarr["base_position"]
    #     valid_base_position = valid_zarr["base_position"]
    #     train_unscaled_leadtimes = train_zarr["leadtime"]
    #     validation_unscaled_leadtimes = valid_zarr["leadtime"]

    if args.dask_array:
        # Load the data from the zarr store using dask array
        train_data = da.from_zarr(zarr_store, component="train/masked_AIX")
        train_target = da.from_zarr(zarr_store, component="train/target_scaled")
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
        train_data = train_zarr["masked_AIX"]
        train_target = train_zarr["target_scaled"]
        valid_data = valid_zarr["masked_AIX"]
        valid_target = valid_zarr["target_scaled"]
        train_leadtimes = train_zarr["leadtime_scaled"]
        validation_leadtimes = valid_zarr["leadtime_scaled"]
        train_base_intensity = train_zarr["base_intensity_scaled"]
        valid_base_intensity = valid_zarr["base_intensity_scaled"]
        train_base_position = train_zarr["base_position"]
        valid_base_position = valid_zarr["base_position"]
        train_unscaled_leadtimes = train_zarr["leadtime"]
        validation_unscaled_leadtimes = valid_zarr["leadtime"]

    # %%
    #  Dataloader & Hyperparameters

    # Let's define some hyperparameters
    batch_size = 128

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

    # %%
    # Calculate the unique leadtimes
    unique_leadtimes = np.unique(train_unscaled_leadtimes)
    # sort the leadtimes from smallest to largest
    unique_leadtimes = np.sort(unique_leadtimes)

    # Create a list of leadtime subsets
    train_leadtime_subsets = [
        unique_leadtimes[: i + 1] for i in range(len(unique_leadtimes))
    ]
    val_leadtime_subsets = [
        unique_leadtimes[: i + 1] for i in range(len(unique_leadtimes))
    ]

    # Create a dictionary of indices for each leadtime subset. The indices are in integer format
    train_leadtime_indices = [
        np.where(np.isin(train_unscaled_leadtimes, leadtime_subset))[0]
        for leadtime_subset in train_leadtime_subsets
    ]
    valid_leadtime_indices = [
        np.where(np.isin(validation_unscaled_leadtimes, leadtime_subset))[0]
        for leadtime_subset in val_leadtime_subsets
    ]
    leadtime_indices = list(zip(train_leadtime_indices, valid_leadtime_indices))

    # %%
    # Make the first datasets & loaders

    initial_idxs = leadtime_indices[0]

    print("Creating validation DaskDataset and dataloader...", flush=True)
    validation_dataset = mlf.ZarrDatasetv2(
        AI_X=valid_data[initial_idxs[1]],
        base_int=valid_base_intensity[initial_idxs[1]],
        target_data=valid_target[initial_idxs[1]],
        device=calc_device,
        num_workers=1,
        track=valid_base_position[initial_idxs[1]],
        leadtimes=validation_leadtimes[initial_idxs[1]],
    )
    # %%
    validation_loader = mlf.make_dataloader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        # pin_memory=True,
    )
    # %%
    print("Creating training DaskDataset and dataloader...", flush=True)
    train_dataset = mlf.ZarrDatasetv2(
        AI_X=train_data[initial_idxs[0]],
        base_int=train_base_intensity[initial_idxs[0]],
        target_data=train_target[initial_idxs[0]],
        device=calc_device,
        num_workers=1,
        track=train_base_position[initial_idxs[0]],
        leadtimes=train_leadtimes[initial_idxs[0]],
    )
    # %%
    train_loader = mlf.make_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        # pin_memory=True,
    )

    # %% Define model and its training hyperparameters
    if args.algorithm == "CNN":
        modelClass = baselines.RegularizedCNN
    elif args.algorithm == "SimpleCNN":
        modelClass = baselines.SimpleCNN
    elif args.algorithm == "UNet":
        modelClass = baselines.UNet
    else:
        raise ValueError("Model not recognized.")

    print(f"Training {modelClass} model...", flush=True)
    # %%
    #  Model
    # We begin by instantiating our baseline model
    CNN = modelClass(
        deterministic=True if args.mode == "deterministic" else False,
        num_scalars=train_dataset.num_scalars,
        input_cols=5 - len(json.loads(args.ablate_cols)),
        cnn_widths=json.loads(args.cnn_width),
        fc_width=args.fc_width,
        dropout=args.dropout,
        dropout2d=args.dropout,
        aux_loss=args.aux_loss,
    )
    CNN.to(calc_device)
    CNN.tags = [str(CNN), args.ai_model, args.mode, args.dropout]
    CNN.savetag = f"{args.ai_model}-{args.mode}"
    CNN.kwargs = args

    optimizer = torch.optim.Adam(CNN.parameters(), lr=1e-4, weight_decay=args.l1_reg)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10
    # )

    num_epochs = 30
    patience = 5  # stop if validation loss increases for patience epochs
    bias_threshold = 30  # stop if validation loss / train loss > bias_threshold

    # %%
    training_curves = {}
    for idx, idx_set in enumerate(leadtime_indices):
        if idx > 1:
            # We then instantiate the DaskDataset class for the training and validation sets.
            # Validation first because it's smaller and will be used to evaluate if the code
            # is working as expected
            val_idxs = idx_set[1]
            train_idxs = idx_set[0]
            print("Creating validation DaskDataset and dataloader...", flush=True)
            validation_dataset = mlf.ZarrDatasetv2(
                AI_X=valid_data[val_idxs],
                base_int=valid_base_intensity[val_idxs],
                target_data=valid_target[val_idxs],
                device=calc_device,
                num_workers=1,
                track=valid_base_position[val_idxs],
                leadtimes=validation_leadtimes[val_idxs],
            )
            validation_loader = mlf.make_dataloader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1,
                # pin_memory=True,
            )

            print("Creating training DaskDataset and dataloader...", flush=True)
            train_dataset = mlf.ZarrDatasetv2(
                AI_X=train_data[train_idxs],
                base_int=train_base_intensity[train_idxs],
                target_data=train_target[train_idxs],
                device=calc_device,
                num_workers=1,
                track=train_base_position[train_idxs],
                leadtimes=train_leadtimes[train_idxs],
            )

            train_loader = mlf.make_dataloader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                # pin_memory=True,
            )

        #  Training
        train_losses = []
        val_losses = []
        if args.aux_loss:
            train_base_losses = []
            val_base_losses = []
            train_aux_losses = []
            val_aux_losses = []

        print("Model Training started...", flush=True)
        start_time = time.strftime("%m-%d-%Hh%M")
        for epoch in range(num_epochs):
            train_loss = 0
            if args.aux_loss:
                train_base_loss = 0
                train_aux_loss = 0
            i = 0
            print("\nTraining:", flush=True)
            for AI_X, scalars, target in train_loader:
                i += 1
                optimizer.zero_grad()

                # convert to float32
                AI_X = AI_X.float()
                scalars = scalars.float()
                target = target.float()

                AI_X = AI_X.to(calc_device)
                scalars = scalars.to(calc_device)
                target = target.to(calc_device)
                # print(AI_X.device, scalars.device)

                prediction = CNN(x=AI_X, scalars=scalars)

                if args.aux_loss:
                    base_loss = loss_func(
                        prediction[:, : CNN.target_size],
                        target,
                    )
                    aux_loss = loss_func(
                        prediction[:, CNN.target_size :],
                        base_loss,
                    )
                    batch_loss = base_loss + aux_loss

                else:
                    batch_loss = loss_func(prediction, target)

                batch_loss.backward()

                optimizer.step()

                train_loss += batch_loss.item()
                if args.aux_loss:
                    train_base_loss += base_loss.item()
                    train_aux_loss += aux_loss.item()

                # print a simple progress bar that shows a dot for each percent of the batch
                print(
                    (
                        "\r"
                        + f"{i}/{len(train_loader)}"
                        + "." * int(20 * i / len(train_loader))
                        + f" Batch loss: {batch_loss.item():.4f}"
                        + (
                            ""
                            if not args.aux_loss
                            else f" Base loss: {base_loss.item():.4f}, Aux loss: {aux_loss.item():.4f}"
                        )
                    ),
                    end="",
                    flush=True,
                )
            # scheduler.step()

            train_loss = train_loss / len(train_loader)
            train_losses.append(train_loss)
            if args.aux_loss:
                train_base_loss = train_base_loss / len(train_loader)
                train_aux_loss = train_aux_loss / len(train_loader)
                train_base_losses.append(train_base_loss)
                train_aux_losses.append(train_aux_loss)

            print("\nValidation:", flush=True)
            val_loss = 0
            if args.aux_loss:
                val_base_loss = 0
                val_aux_loss = 0
            i = 0
            with torch.no_grad():
                for AI_X, scalars, target in validation_loader:
                    # print(
                    #     f"\r {AI_X.dtype} {scalars.dtype} {target.dtype}",
                    #     end="",
                    #     flush=True,
                    # )
                    i += 1
                    # batch_data = AI_X.to(calc_device)
                    # batch_scalars = scalars.to(calc_device)
                    # prediction = CNN(
                    #     x=batch_data,
                    #     scalars=batch_scalars,
                    # )

                    AI_X = AI_X.float()
                    scalars = scalars.float()
                    target = target.float()

                    AI_X = AI_X.to(calc_device)
                    scalars = scalars.to(calc_device)
                    target = target.to(calc_device)
                    prediction = CNN(x=AI_X, scalars=scalars)

                    if args.aux_loss:
                        base_loss = loss_func(
                            prediction[:, : CNN.target_size],
                            target,
                        )
                        aux_loss = loss_func(
                            prediction[:, CNN.target_size :],
                            base_loss,
                        )
                        batch_loss = base_loss + aux_loss
                    else:
                        batch_loss = loss_func(prediction, target)

                    val_loss += batch_loss.item()
                    if args.aux_loss:
                        val_base_loss += base_loss.item()
                        val_aux_loss += aux_loss.item()

                    # print a simple progress bar that shows a dot for each percent of the batch
                    print(
                        (
                            "\r"
                            + f"{i}/{len(validation_loader)}"
                            + "." * int(20 * i / len(validation_loader))
                            + (
                                ""
                                if not args.aux_loss
                                else f" Base loss: {base_loss.item():.4f}, Aux loss: {aux_loss.item():.4f}"
                            )
                        ),
                        end="",
                        flush=True,
                    )

            val_loss = val_loss / len(validation_loader)
            val_losses.append(val_loss)
            if args.aux_loss:
                val_base_loss = val_base_loss / len(validation_loader)
                val_aux_loss = val_aux_loss / len(validation_loader)
                val_base_losses.append(val_base_loss)
                val_aux_losses.append(val_aux_loss)

            # Save if the validation loss is the best so far
            if val_loss <= max(val_losses):
                # torch.save(
                #     CNN,
                #     os.path.join(
                #         result_dir,
                #         f"best_model_{str(CNN)}_{start_time}_epoch-{epoch+1}_{args.ai_model}_{args.mode}_{args.cnn_width}.pt",
                #     ),
                # )
                torch.save(
                    CNN,
                    os.path.join(
                        result_dir,
                        f"{str(CNN.tags[0])}_{idx}_{start_time}_epoch-{epoch+1}_{CNN.savetag}{args.tag}.pt",
                    ),
                )

            print(
                f"\nEpoch: {epoch+1}/{num_epochs},",
                f"train_loss: {train_loss},",
                f"val_loss: {val_loss},",
                flush=True,
                sep=" ",
            )

            # Early stopping
            # Stop if the validation loss has been increasing on average for the last patience epochs
            if epoch > patience:
                if (
                    np.mean(np.gradient(val_losses)[-patience:]) > 0
                    or val_loss / train_loss > bias_threshold
                ):
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # Let's save the train and validation losses in a pickled dictionary
            losses = {"train": train_losses, "validation": val_losses}
            if args.aux_loss:
                losses["train_base"] = train_base_losses
                losses["train_aux"] = train_aux_losses
                losses["validation_base"] = val_base_losses
                losses["validation_aux"] = val_aux_losses

            with open(
                os.path.join(
                    result_dir,
                    f"{CNN.tags[0]}_losses_{start_time}_step-{idx}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(losses, f)

        # plot the learning curves
        fig, ax = plt.subplots()
        ax.plot(
            train_losses,
            label="Train Loss",
            color=np.array([27, 166, 166]) / 255,
            alpha=0.8,
        )
        ax.plot(
            val_losses,
            label="Validation Loss",
            color=np.array([191, 6, 92]) / 255,
            alpha=0.8,
        )
        ax.set_xlabel("Epoch")
        tick_space = len(train_losses) // 10
        ax.xaxis.set_ticks(
            np.arange(0, len(train_losses), tick_space if tick_space > 1 else 1)
        )
        ax.set_ylabel(f"Loss: {str(loss_func)}")
        ax.legend()
        toolbox.plot_facecolors(fig=fig, axes=ax)
        # fig.savefig(
        #     os.path.join(
        #         result_dir, f"CNN_{str(CNN)}_{args.ai_model}_losses_{start_time}.png"
        #     )
        # )
        fig.savefig(
            os.path.join(
                result_dir, f"{CNN.tags[0]}_losses_{start_time}_step-{idx}.png"
            )
        )

    # %%
