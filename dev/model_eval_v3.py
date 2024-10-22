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

    # Check if the cache directory includes a zarray store for the data
    try:
        zarr_path = os.path.join(cache_dir, "zarray_store")
        zarr_store = zr.DirectoryStore(zarr_path)
    except FileNotFoundError:
        raise (
            "Zarray Cache directory not found. Evaluation requires that the data be cached. Please run the data preparation script first."
        )

    root = zr.open_group(zarr_path)

    # Check that the validation group exists
    assert (
        "validation" in root
    ), "Validation group not found in cache directory. Evaluation is run on the validation set."

    # Check that the validationmask exists if --mask is True
    if args.mask:
        assert (
            "validation/mask" in root
        ), "Validation mask not found in cache directory. Please run the data preparation script with --mask True."

    # Check that the validation leadtime group exists
    assert (
        "validation/leadtime" in root
    ), "Validation leadtime group not found in cache directory. Evaluation is run on the validation set."

    # Load the data from the zarr store
    train_target = zr.open(zarr_store, mode="r")["train/target_scaled"]
    valid_target = zr.open(zarr_store, mode="r")["validation/target_scaled"]
    leadtimes = zr.open(zarr_store, mode="r")["train/leadtime"]

    unique_leads = np.unique(leadtimes)

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

    # We then instantiate the DaskDataset class for the training and validation sets.
    # Validation first because it's smaller and will be used to evaluate if the code
    # is working as expected
    print("Creating validation DaskDataset and dataloader...", flush=True)
    validation_dataset = mlf.ZarrDataset(
        AI_X=valid_data,
        base_int=valid_base_intensity,
        target_data=valid_target,
        device=calc_device,
        num_workers=num_cores,
        track=valid_base_position,
        leadtimes=validation_leadtimes,
    )

    validation_loader = mlf.make_dataloader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
    )

    print("Creating training DaskDataset and dataloader...", flush=True)
    train_dataset = mlf.ZarrDataset(
        AI_X=train_data,
        base_int=train_base_intensity,
        target_data=train_target,
        device=calc_device,
        num_workers=num_workers,
        track=train_base_position,
        leadtimes=train_leadtimes,
    )
    train_loader = mlf.make_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True,
    )

    if emulate:
        input("Press Enter to continue...")
    #  Model
    # We begin by instantiating our baseline model
    CNN = baselines.RegularizedCNN(
        deterministic=True if args.mode == "deterministic" else False,
        num_scalars=train_dataset.num_scalars,
        input_cols=5 - len(json.loads(args.ablate_cols)),
        cnn_widths=json.loads(args.cnn_width),
        fc_width=args.fc_width,
        dropout=args.dropout,
        dropout2d=args.dropout,
        aux_loss=args.aux_loss,
    ).to(calc_device)
    # CNN = baselines.SimpleCNN(
    #     deterministic=True if args.mode == "deterministic" else False,
    #     num_scalars=train_dataset.num_scalars,
    #     input_cols=5 - len(json.loads(args.ablate_cols)),
    #     cnn_widths=json.loads(args.cnn_width),
    #     fc_width=args.fc_width,
    #     aux_loss=args.aux_loss,
    # ).to(calc_device)

    optimizer = torch.optim.Adam(CNN.parameters(), lr=1e-4)  # , weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10
    )

    num_epochs = 100
    patience = 5  # stop if validation loss increases for patience epochs
    bias_threshold = 30  # stop if validation loss / train loss > bias_threshold

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

            # batch_data = AI_X.to(calc_device)
            # batch_scalars = scalars.to(calc_device)
            # prediction = CNN(
            #     x=batch_data,
            #     scalars=batch_scalars,
            # )
            # print(AI_X.device, scalars.device)
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
        scheduler.step()

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
            torch.save(
                CNN,
                os.path.join(
                    result_dir,
                    f"best_model_{str(CNN)}_{start_time}_epoch-{epoch+1}_{args.ai_model}_{args.mode}_{args.cnn_width}.pt",
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
                f"CNN_{str(CNN)}_losses_{start_time}_{args.ai_model}_{args.mode}_{args.cnn_width}.pkl",
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
        fig.savefig(
            os.path.join(
                result_dir, f"CNN_{str(CNN)}_{args.ai_model}_losses_{start_time}.png"
            )
        )

    # %%

    # def eval_prediction(
    #     model,  # ML model
    #     loader,  # Dataloader
    #     target_scaler,  # Target scaler
    #     baseline_pred,  # Baseline prediction
    #     y_true,  # Ground truth
    #     result_dir,  # Results directory
    #     start_time,  # Start time
    #     set_type,  # Type of dataset
    #     data,  # Data dictionary
    #     **kwargs,
    # ):
    #     # Predict the validation data with the CNN model
    #     with torch.no_grad():
    #         y_hat = None
    #         for AI_X, base_int, target in loader:
    #             pred = model(AI_X, base_int)
    #             if y_hat is None:
    #                 y_hat = pred.cpu().numpy()
    #             else:
    #                 y_hat = np.concatenate([y_hat, pred.cpu().numpy()])

    #     # reverse transform y_hat
    #     y_hat = target_scaler.inverse_transform(y_hat)

    #     #  Compute y_true, y_baseline
    #     y_baseline = baseline_pred.compute()

    #     #  Evaluation
    #     # We will want to evaluate the model using the Root Mean Squared Error and
    #     # the Mean Absolute Error, as well as the associated skill scores compared
    #     # to the persistence model

    #     global_performance = metrics.summarize_performance(
    #         y_true,  # Ground truth
    #         y_hat,  # Model prediction
    #         y_baseline,  # Baseline, used for skill score calculations
    #         [root_mean_squared_error, mean_absolute_error],
    #     )

    #     # Plotting Global Performance
    #     fig, axes = plt.subplots(
    #         1, 2, figsize=(15, 5), dpi=150, gridspec_kw={"width_ratios": [2, 1]}
    #     )
    #     metrics.plot_performance(
    #         global_performance,
    #         axes,
    #         model_name=f"{str(CNN)}",
    #         baseline_name=kwargs.get("baseline_name", "Persistence"),
    #     )
    #     toolbox.plot_facecolors(fig=fig, axes=axes)
    #     # Save the figure in the results directory
    #     fig.savefig(
    #         os.path.join(
    #             result_dir,
    #             f"CNN_global_performance_{str(model)}_{start_time}_{set_type}.png",
    #         )
    #     )

    #     # Per leadtime analysis
    #     unique_leads = data[set_type]["leadtime"].compute()
    #     num_leads = len(unique_leads)
    #     # set up the axes for the lead time plots

    #     fig, axes = plt.subplots(
    #         num_leads,  # One row per lead time
    #         2,
    #         figsize=(15, 5 * num_leads),
    #         dpi=150,
    #         gridspec_kw={"width_ratios": [2, 1]},
    #     )
    #     for idx, lead in enumerate(unique_leads):
    #         lead_mask = data[set_type]["leadtime"] == lead

    #         y_true_lead = y_true[lead_mask]
    #         y_hat_lead = y_hat[lead_mask]
    #         y_baseline_lead = y_baseline[lead_mask]

    #         lead_performance = metrics.summarize_performance(
    #             y_true_lead,  # Ground truth
    #             y_hat_lead,  # Model prediction
    #             y_baseline_lead,  # Baseline, used for skill score calculations
    #             [root_mean_squared_error, mean_absolute_error],
    #         )

    #         metrics.plot_performance(
    #             lead_performance,
    #             axes[idx],
    #             model_name=f"{str(model)}",
    #             baseline_name="Persistence",
    #         )
    #         toolbox.plot_facecolors(fig=fig, axes=axes[idx])

    #         # Append the lead time to the title for both axes
    #         axes[idx][0].set_title(
    #             f"\n Lead Time: +{lead}h \n" + axes[idx][0].get_title()
    #         )
    #         axes[idx][1].set_title(
    #             f"\n Lead Time: +{lead}h \n" + axes[idx][1].get_title()
    #         )

    #     # Save the figure in the results directory
    #     fig.savefig(
    #         os.path.join(
    #             result_dir,
    #             f"CNN_lead_performance_{str(model)}_{start_time}_{set_type}.png",
    #         )
    #     )

    # # Evaluate on the validation set

    # # We want to compare the model to a simple baseline, in this case
    # # the persistence model. Since our target is delta, persistence
    # # is simply 0
    # y_persistence = np.zeros_like(valid_delta)

    # #  Evaluation
    # # Let's start by loading the validation outputs into a variable
    # # for easier access
    # y_true = valid_delta

    # eval_prediction(
    #     model=CNN,
    #     loader=validation_loader,
    #     target_scaler=target_scaler,
    #     baseline_pred=y_persistence,
    #     y_true=y_true,
    #     result_dir=result_dir,
    #     start_time=start_time,
    #     set_type="validation",
    #     data=data,
    # )
    # # Evaluate on the training set

    # # We want to compare the model to a simple baseline, in this case
    # # the persistence model. Since our target is delta, persistence
    # # is simply 0
    # y_persistence = np.zeros_like(train_delta)

    # #  Evaluation
    # # Let's start by loading the validation outputs into a variable
    # # for easier access
    # y_true = train_delta

    # eval_prediction(
    #     model=CNN,
    #     loader=train_loader,
    #     target_scaler=target_scaler,
    #     baseline_pred=y_persistence,
    #     y_true=y_true,
    #     result_dir=result_dir,
    #     start_time=start_time,
    #     set_type="train",
    #     data=data,
    # )
