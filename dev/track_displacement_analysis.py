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

# import torch
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
            # "--verbose",
        ]

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

    args = parser.parse_args()

    print("Imports successful", flush=True)

    #  Setup
    datadir = args.datadir
    cache_dir = args.cache_dir + f"_{args.ai_model}"
    result_dir = args.result_dir
    dask.config.set(scheduler="threads")

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
            use_cached=not args.overwrite_cache,
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
    # %%
    unique_leadtimes = da.unique(data["train"]["leadtime"][train_ldt_mask]).compute()
    unique_leadtimes = np.sort(unique_leadtimes).astype(int)

    # %%
    leadtimes = []
    displacements = []
    delta_lons = []
    delta_lats = []
    mean_displacement = []
    mean_delta_lon = []
    mean_delta_lat = []
    displacement_percentiles = []
    delta_lon_percentiles = []
    delta_lat_percentiles = []

    q = [1, 5, 16, 25, 50, 75, 84, 95, 99]

    # %%

    for unique_leadtime in unique_leadtimes:
        print(f"Unique leadtime: {unique_leadtime}")
        distance, delta_lon, delta_lat = toolbox.analyze_displacement_hist(
            storm_set=sets["train"], leadtime=int(unique_leadtime)
        )
        leadtimes.append(unique_leadtime)
        displacements.append(distance)
        delta_lons.append(delta_lon)
        delta_lats.append(delta_lat)
        mean_displacement.append(np.mean(distance))
        mean_delta_lon.append(np.mean(delta_lon))
        mean_delta_lat.append(np.mean(delta_lat))
        displacement_percentiles.append(np.percentile(distance, q=q))
        delta_lon_percentiles.append(np.percentile(delta_lon, q=q))
        delta_lat_percentiles.append(np.percentile(delta_lat, q=q))

    # %%
    # Transform the lists into numpy arrays
    leadtimes = np.array(leadtimes)
    # displacements = np.array(displacements)
    # delta_lons = np.array(delta_lons)
    # delta_lats = np.array(delta_lats)
    mean_displacement = np.array(mean_displacement)
    mean_delta_lon = np.array(mean_delta_lon)
    mean_delta_lat = np.array(mean_delta_lat)
    displacement_percentiles = np.array(displacement_percentiles)
    delta_lon_percentiles = np.array(delta_lon_percentiles)
    delta_lat_percentiles = np.array(delta_lat_percentiles)

    # Calculate the average speed for each leadtime
    speed = mean_displacement / np.array(leadtimes)
    mean_speed = np.mean(speed)

    lon_speed = mean_delta_lon / np.array(leadtimes)
    mean_lon_speed = np.mean(lon_speed)

    lat_speed = mean_delta_lat / np.array(leadtimes)
    mean_lat_speed = np.mean(lat_speed)

    # %%

    # Define colorblind safe colors for plotting
    colors = [
        np.array([215, 166, 122]) / 255,
        np.array([0, 148, 199]) / 255,
        np.array([214, 7, 114]) / 255,
        np.array([50, 50, 50]) / 255,
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15), dpi=150)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot(
            leadtimes,
            displacement_percentiles[:, i],
            label=f"{q[i]}th percentile",
            color=colors[2],
        )

        # plot the line defined by the mean speed
        x = np.linspace(0, 168, 100)
        y = mean_speed * x

        ax.plot(x, y, color=colors[3], label="Displacement at Mean Speed")

        ax.set_title(f"{q[i]}th Percentile Displacements")
        ax.set_xlabel("Leadtime (hours)")
        ax.set_ylabel("Displacement (km)")
        ax.legend()
    toolbox.plot_facecolors(fig=fig, axes=axes)

    # %%
    # Make a 3D surface where the x-axis is the leadtime, the y-axis is the percentile, and the z-axis is the displacement
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # q = [1, 5, 25, 50, 75, 95, 99]
    # q = np.tile(q, (displacement_percentiles.shape[0], 1))
    X, Y = np.meshgrid(leadtimes, q)
    Z = displacement_percentiles.T
    ax.plot_surface(X, Y, Z, alpha=1, cmap="viridis", label="Displacement Percentiles")

    # plot the plane defined by the mean speed
    x = np.linspace(0, 168, 100)
    y = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = mean_speed * X
    ax.plot_surface(
        X,
        Y,
        Z,
        alpha=1,
        color=np.array([200, 50, 50]) / 255,
        label="Displacement at Mean Speed",
    )

    # %% Logarithmic fitting for 99th percentile

    from scipy.optimize import curve_fit

    def log_func(x, a, b, c, d):
        return a * np.log(b * x + c) + d

    tgt_idx = -3

    # Fit the leadtimes and displacement percentiles to the logarithmic function
    popt, pcov = curve_fit(log_func, leadtimes, displacement_percentiles[:, tgt_idx])

    np.save(
        "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/log_params_84ptile.npy",
        popt,
    )

    # Plot the fitted curve
    fig, ax = plt.subplots()
    ax.scatter(
        leadtimes,
        displacement_percentiles[:, tgt_idx].squeeze(),
        c=colors[3],
        label="Data",
        # s=5,
        marker="+",
    )
    plot_x = np.linspace(0, leadtimes.max() + 1, 100)
    ax.plot(
        plot_x,
        log_func(plot_x, *popt),
        c=colors[2],
        label=rf"$F(t_{{ld}}) = {popt[0]:.2f} \cdot \log({popt[1]:.2f} \cdot t_{{ld}} + {popt[2]:.2f}){popt[3]:.2f}$",
    )
    ax.set_xlabel("Leadtime (hours)")
    ax.set_ylabel("Displacement (km)")
    ax.legend()
    fig.suptitle(
        f"{q[tgt_idx]}th Percentile Displacement vs. Leadtime",
        color=np.array([242, 240, 228]) / 255,
    )
    toolbox.plot_facecolors(fig=fig, axes=ax)

    # %%

    # # Calculate the intersection points
    # intersection_points = []
    # for i in range(len(leadtimes)):
    #     for j in range(len(q)):
    #         if np.isclose(Z[j, i], mean_speed * leadtimes[i], atol=1):
    #             intersection_points.append((leadtimes[i], q[j], Z[j, i]))

    # # Extract the intersection points for plotting
    # if intersection_points:
    #     intersection_points = np.array(intersection_points)
    #     ax.plot(
    #         intersection_points[:, 0],
    #         intersection_points[:, 1],
    #         intersection_points[:, 2],
    #         color="g",
    #         marker="o",
    #         linestyle="None",
    #         label="Intersection Points",
    #     )

    # ax.set_xlabel("Leadtime (hours)")
    # ax.set_ylabel("Percentile")
    # ax.set_zlabel("Displacement (km)")

    # plt.legend()
    # plt.show()

# %%
# Define x-values
x_straight = np.linspace(0, 30, 100)
x_curves = np.linspace(30, 42, 100)

# Define y-values for the straight line
y_straight = np.ones_like(x_straight)


# Define functions for the curves
def quadratic_convex_down(x):
    return 1 - ((x - 30) / 12) ** 2


def straight_line(x):
    return 1 - (x - 30) / 12


def exponential_decline(x):
    return np.exp(-0.3 * (x - 30))


# Calculate y-values for the curves
y_quadratic_convex_down = quadratic_convex_down(x_curves)
y_straight_line = straight_line(x_curves)
y_exp_decline = exponential_decline(x_curves)

# Create the plot using fig, ax API
fig, ax = plt.subplots(figsize=(10, 2), dpi=150)

# Plot the straight line
ax.plot(
    x_straight, y_straight, label=r"mask=1 (0 to $mask_{rad_{min}}$)", color=colors[3]
)

# Plot the curves
ax.plot(
    x_curves, y_quadratic_convex_down, label="Quadratic (Convex Down)", color=colors[0]
)
ax.plot(x_curves, y_straight_line, label="Straight Line", color=colors[1])
ax.plot(x_curves, y_exp_decline, label="Exponential Decline", color=colors[2])

# Add labels and title
ax.set_xlabel("radius")
ax.set_ylabel("Mask Value")
ax.set_title("Proposed Decay Strategies")
ax.legend(fontsize=8)

# Customize x-axis ticks and labels
ax.set_xticks([0, 30, 40])
ax.set_xticklabels(["", r"$mask_{rad_{(min)}}$", r"$mask_{rad_{(min)}} + fade_{rad}$"])
ax.set_xlim(0, 42)
ax.set_ylim(0, 1.2)
# Show the plot
ax.grid(True)

toolbox.plot_facecolors(fig=fig, axes=ax)
plt.tight_layout()
plt.show()
# %%
