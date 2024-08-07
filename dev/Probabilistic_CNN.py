# %% This line makes it run on jupyter notebook by default in vscode
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

# from dask import optimize
import time


# Backend Libraries
import joblib as jl

from utils import toolbox, ML_functions as mlf
import metrics

# Importing the sklearn metrics
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

if __name__ == "__main__":
    print("Imports successful", flush=True)

    # check if the context has been set for torch multiprocessing
    if not torch.multiprocessing.get_start_method(allow_none=True) == "spawn":
        torch.multiprocessing.set_start_method("spawn")

    #  Setup
    datadir = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha"
    cache_dir = "/scratch/mgomezd1/cache"
    result_dir = (
        "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/"
    )

    use_cuda = False
    # Check for GPU availability
    if torch.cuda.device_count() > 0 and use_cuda:
        calc_device = torch.device("cuda:0")
    else:
        calc_device = torch.device("cpu")
        # # Ask the user if they want to continue even though there's no GPU
        # print("No GPU available, continue with CPU? (y/n)")
        # if input() != "y":
        #     sys.exit()

    num_cores = int(subprocess.check_output(["nproc"], text=True).strip())

    #  Data Loading
    rng = np.random.default_rng(seed=42)

    years = list(range(2013, 2020))
    rng.shuffle(years)

    print("Loading datasets...", flush=True)

    sets, data = toolbox.get_ai_sets(
        {
            "train": years[:-2],
            "validation": years[-2:],
            "test": [2020],
        },
        datadir=datadir,
        test_strategy="custom",
        # use_cached=False,
        # verbose=True,
        # debug=True,
    )

    # sets, data = toolbox.get_ai_sets(
    #     {"train": [2014],"validation": [2015], }, # "test": [2020]},
    #     datadir=datadir,
    #     test_strategy="custom",
    #     # use_cached=False,
    #     # verbose=True,
    #     # debug=True,
    # )

    #  Reload for interactive development
    try:
        import importlib

        importlib.reload(baselines)
        importlib.reload(metrics)
        importlib.reload(mlf)
    except:
        import baselines

    #  Preprocessing
    # We will want to normalize the inputs for the model
    # to work properly. We will use the AI_StandardScaler for this
    # purpose
    AI_scaler = None
    from_cache = True

    fpath = os.path.join(cache_dir, "AI_scaler.pkl")

    if from_cache:
        # print('Loading AI scaler from cache...', flush=True)
        if os.path.exists(fpath):
            print("Loading AI scaler from cache...", flush=True)
            with open(fpath, "rb") as f:
                AI_scaler = pickle.load(f)

    if AI_scaler is None:
        print("Fitting AI datascaler...", flush=True)
        AI_data = data["train"]["inputs"]
        # AI_data = optimize(AI_data)[0]
        AI_scaler = mlf.AI_StandardScaler()
        AI_scaler.fit(AI_data, num_workers=num_cores)

        # save the scaler to the cache
        with open(fpath, "wb") as f:
            pickle.dump(AI_scaler, f)

    # We also want to train a scaler for the base intensity
    print("Fitting base intensity scaler...", flush=True)
    base_scaler = StandardScaler()
    base_scaler.fit(data["train"]["base_intensity"])

    # We also want to precalculate the delta intensity for the
    # training and validation sets
    print("Calculating target (i.e., delta intensities)...", flush=True)
    train_delta = data["train"]["outputs"] - data["train"]["base_intensity"]
    valid_delta = data["validation"]["outputs"] - data["validation"]["base_intensity"]

    # And make a scaler for the target data
    target_scaler = StandardScaler()
    target_scaler.fit(train_delta)

    #  Dataloader & Hyperparameters

    # Let's define some hyperparameters
    batch_size = 32
    loss_func = metrics.CRPS_ML
    num_workers = (
        int(num_cores * 2 / 3)
        if (calc_device == torch.device("cpu") and num_cores > 1)
        else num_cores
    )
    dask.config.set(scheduler="synchronous")

    # We then instantiate the DaskDataset class for the training and validation sets
    print("Creating training DaskDataset...", flush=True)
    train_dataset = mlf.DaskDataset(
        AI_X=data["train"]["inputs"],
        AI_scaler=AI_scaler,
        base_int=data["train"]["base_intensity"],
        base_scaler=base_scaler,
        target_data=train_delta,
        target_scaler=target_scaler,
        device=calc_device,
        cachedir="/scratch/mgomezd1/cache",  # os.path.join(datadir, 'cache'),
        zarr_name="train",
        overwrite=False,
        num_workers=num_cores,
        # load_into_memory=True,
    )

    print("Creating validation DaskDataset...", flush=True)
    validation_dataset = mlf.DaskDataset(
        AI_X=data["validation"]["inputs"],
        AI_scaler=AI_scaler,
        base_int=data["validation"]["base_intensity"],
        base_scaler=base_scaler,
        target_data=valid_delta,
        target_scaler=target_scaler,
        device=calc_device,
        cachedir="/scratch/mgomezd1/cache",  # os.path.join(datadir, 'cache'),
        zarr_name="validation",
        overwrite=False,
        num_workers=num_cores,
        # load_into_memory=True,
    )

    # And make the respective dataloaders
    print("Creating dataloaders...", flush=True)
    train_loader = mlf.make_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    validation_loader = mlf.make_dataloader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    #  Model
    # We begin by instantiating our baseline model
    # CNN = baselines.TC_DeltaIntensity_CNN(deterministic=True).to(calc_device)
    # CNN = baselines.Regularized_Dilated_CNN(
    #     deterministic=True, dropout=0.05, dropout2d=0.05
    # ).to(calc_device)
    # CNN = baselines.TC_DeltaIntensity_nonDilCNN(deterministic=False).to(calc_device)
    CNN = baselines.Regularized_NonDil_CNN(
        deterministic=True, dropout=0.1, dropout2d=0.1
    ).to(calc_device)

    optimizer = torch.optim.Adam(CNN.parameters(), lr=1e-4)  # , weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10
    )

    num_epochs = 100
    patience = 4  # stop if validation loss increases for patience epochs
    bias_threshold = 20  # stop if validation loss / train loss > bias_threshold

    #  Training
    train_losses = []
    val_losses = []

    print("Model Training started...", flush=True)
    start_time = time.strftime("%m-%d-%Hh%M")
    for epoch in range(num_epochs):
        train_loss = 0
        i = 0
        print("\nTraining:", flush=True)
        for AI_X, base_int, target in train_loader:
            i += 1
            optimizer.zero_grad()

            prediction = CNN(AI_X, base_int)
            batch_loss = loss_func(prediction, target)

            batch_loss.backward()

            optimizer.step()

            train_loss += batch_loss.item()

            # print a simple progress bar that shows a dot for each percent of the batch
            print(
                "\r"
                + f"{i}/{len(train_loader)}"
                + "." * int(20 * i / len(train_loader))
                + f" Batch loss: {batch_loss.item()}",
                end="",
                flush=True,
            )
        scheduler.step()

        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

        print("\nValidation:", flush=True)
        val_loss = 0
        i = 0
        with torch.no_grad():
            for AI_X, base_int, target in validation_loader:
                i += 1
                prediction = CNN(AI_X, base_int)
                batch_loss = loss_func(prediction, target)

                val_loss += batch_loss.item()

                # print a simple progress bar that shows a dot for each percent of the batch
                print(
                    "\r"
                    + f"{i}/{len(validation_loader)}"
                    + "." * int(20 * i / len(validation_loader)),
                    end="",
                    flush=True,
                )

        val_loss = val_loss / len(validation_loader)
        val_losses.append(val_loss)

        # Save if the validation loss is the best so far
        if val_loss <= max(val_losses):
            torch.save(
                CNN,
                os.path.join(
                    result_dir, f"best_model_{str(CNN)}_{start_time}_epoch-{epoch+1}.pt"
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
        # Stop if the validation loss is greater than all previous patience epochs
        if epoch > patience:
            if (
                all(val_loss > x for x in val_losses[-patience:])
                or val_loss / train_loss > bias_threshold
            ):
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Let's save the train and validation losses in a pickled dictionary
        losses = {"train": train_losses, "validation": val_losses}
        with open(
            os.path.join(result_dir, f"CNN_{str(CNN)}_losses_{start_time}.pkl"), "wb"
        ) as f:
            pickle.dump(losses, f)

    # %%

    def eval_prediction(
        model,  # ML model
        loader,  # Dataloader
        target_scaler,  # Target scaler
        baseline_pred,  # Baseline prediction
        y_true,  # Ground truth
        result_dir,  # Results directory
        start_time,  # Start time
        set_type,  # Type of dataset
        data,  # Data dictionary
        **kwargs,
    ):
        # Predict the validation data with the CNN model
        with torch.no_grad():
            y_hat = None
            for AI_X, base_int, target in loader:
                pred = model(AI_X, base_int)
                if y_hat is None:
                    y_hat = pred.cpu().numpy()
                else:
                    y_hat = np.concatenate([y_hat, pred.cpu().numpy()])

        # reverse transform y_hat
        y_hat = target_scaler.inverse_transform(y_hat)
        y_mean = y_hat[:, ::2]
        y_std = y_hat[:, 1::2]
        y_lower = y_mean - y_std
        y_upper = y_mean + y_std

        #  Compute y_true, y_baseline
        y_baseline = baseline_pred.compute()

        #  Evaluation
        # We will want to evaluate the model using the Root Mean Squared Error and
        # the Mean Absolute Error, as well as the associated skill scores compared
        # to the persistence model

        global_performance = metrics.summarize_performance(
            y_true,  # Ground truth
            y_hat,  # Model prediction
            y_baseline,  # Baseline, used for skill score calculations
            [root_mean_squared_error, mean_absolute_error],
        )

        # Plotting Global Performance
        fig, axes = plt.subplots(
            1, 2, figsize=(15, 5), dpi=150, gridspec_kw={"width_ratios": [2, 1]}
        )
        metrics.plot_performance(
            global_performance,
            axes,
            model_name=f"{str(CNN)}",
            baseline_name=kwargs.get("baseline_name", "Persistence"),
        )
        toolbox.plot_facecolors(fig=fig, axes=axes)
        # Save the figure in the results directory
        fig.savefig(
            os.path.join(
                result_dir,
                f"CNN_global_performance_{str(model)}_{start_time}_{set_type}.png",
            )
        )

        # Per leadtime analysis
        unique_leads = data[set_type]["leadtime"].compute()
        num_leads = len(unique_leads)
        # set up the axes for the lead time plots

        fig, axes = plt.subplots(
            num_leads,  # One row per lead time
            2,
            figsize=(15, 5 * num_leads),
            dpi=150,
            gridspec_kw={"width_ratios": [2, 1]},
        )
        for idx, lead in enumerate(unique_leads):
            lead_mask = data[set_type]["leadtime"] == lead

            y_true_lead = y_true[lead_mask]
            y_hat_lead = y_hat[lead_mask]
            y_baseline_lead = y_baseline[lead_mask]

            lead_performance = metrics.summarize_performance(
                y_true_lead,  # Ground truth
                y_hat_lead,  # Model prediction
                y_baseline_lead,  # Baseline, used for skill score calculations
                [root_mean_squared_error, mean_absolute_error],
            )

            metrics.plot_performance(
                lead_performance,
                axes[idx],
                model_name=f"{str(model)}",
                baseline_name="Persistence",
            )
            toolbox.plot_facecolors(fig=fig, axes=axes[idx])

            # Append the lead time to the title for both axes
            axes[idx][0].set_title(
                f"\n Lead Time: +{lead}h \n" + axes[idx][0].get_title()
            )
            axes[idx][1].set_title(
                f"\n Lead Time: +{lead}h \n" + axes[idx][1].get_title()
            )

        # Save the figure in the results directory
        fig.savefig(
            os.path.join(
                result_dir,
                f"CNN_lead_performance_{str(model)}_{start_time}_{set_type}.png",
            )
        )

    # Evaluate on the validation set

    # We want to compare the model to a simple baseline, in this case
    # the persistence model. Since our target is delta, persistence
    # is simply 0
    y_persistence = np.zeros_like(valid_delta)

    #  Evaluation
    # Let's start by loading the validation outputs into a variable
    # for easier access
    y_true = valid_delta

    eval_prediction(
        model=CNN,
        loader=validation_loader,
        target_scaler=target_scaler,
        baseline_pred=da.from_array(y_persistence),
        y_true=y_true,
        result_dir=result_dir,
        start_time=start_time,
        set_type="validation",
        data=data,
    )
    # Evaluate on the training set

    # We want to compare the model to a simple baseline, in this case
    # the persistence model. Since our target is delta, persistence
    # is simply 0
    y_persistence = np.zeros_like(train_delta)

    #  Evaluation
    # Let's start by loading the validation outputs into a variable
    # for easier access
    y_true = train_delta

    eval_prediction(
        model=CNN,
        loader=train_loader,
        target_scaler=target_scaler,
        baseline_pred=da.from_array(y_persistence),
        y_true=y_true,
        result_dir=result_dir,
        start_time=start_time,
        set_type="train",
        data=data,
    )
