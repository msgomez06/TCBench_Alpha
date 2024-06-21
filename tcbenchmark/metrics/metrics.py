# Climatology Baseline Models
# Date: 2024.05.24
# This file contains the metrics used for evaluating model performance.
# The metrics are implemented following the scikit-learn convention.
# Author: Milton Gomez

# From SKLearn:
# Functions named as ``*_score`` return a scalar value to maximize: the higher
# the better.

# Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
# the lower the better.

# %% Imports
# OS and IO
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dask.array as da
import pandas as pd

try:
    import torch
except:
    print("torch not available")

# Backend Libraries
import joblib as jl

from ..utils import toolbox, constants, ML_functions as mlf
from ..utils import data_lib as dlib

# %% Reference Dictionaries
short_names = {
    "root_mean_squared_error": "RMSE",
    "mean_squared_error": "MSE",
    "mean_absolute_error": "MAE",
    "mean_absolute_percentage_error": "MAPE",
    "mean_squared_logarithmic_error": "MSLE",
    "r2_score": "R2",
    "explained_variance_score": "EV",
    "max_error": "ME",
    "mean_poisson_deviance": "MPD",
    "mean_gamma_deviance": "MGD",
    "mean_tweedie_deviance": "MTD",
    "continuous_ranked_probability_score": "CRPS",
}

units = {"wind": "kts", "pressure": "hPa"}


# %% Utilities
def _check_regression(y_true, y_pred):
    """Check that the regression inputs are of the correct form."""
    raise NotImplementedError
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape."
    # assert y_true.ndim == 1, "y_true and y_pred must be 1D arrays."
    # assert y_pred.ndim == 1, "y_true and y_pred must be 1D arrays."
    return y_true, y_pred


def _check_classification(y_true, y_pred):
    """Check that the classification inputs are of the correct form."""
    raise NotImplementedError
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape."
    # assert y_true.ndim == 1, "y_true and y_pred must be 1D arrays."
    # assert y_pred.ndim == 1, "y_true and y_pred must be 1D arrays."
    return y_true, y_pred


# %% Metrics


def CRPS(y_true, y_pred):
    """Compute the Continuous Ranked Probability Score (CRPS).

    The CRPS is a probabilistic metric that evaluates the accuracy of a
    probabilistic forecast. It is defined as the integral of the squared
    difference between the cumulative distribution function (CDF) of the
    forecast and the CDF of the observations.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The true target values.

    y_pred : array-like of shape (n_samples, n_classes)
        The predicted probabilities for each class.

    Returns
    -------
    crps : float
        The CRPS value.
    """
    # Check that the inputs are of the correct form
    y_true, y_pred = _check_classification(y_true, y_pred)

    # Compute the CRPS
    raise NotImplementedError
    return crps


def CRPS_ML(y_pred, y_true, **kwargs):
    """Compute the Continuous Ranked Probability Score (CRPS).

    The CRPS is a probabilistic metric that evaluates the accuracy of a
    probabilistic forecast. It is defined as the integral of the squared
    difference between the cumulative distribution function (CDF) of the
    forecast and the CDF of the observations.

    Parameters
    ----------

    y_pred : array-like of shape (n_samples, mu, sigma)
        The predicted probability parameters for each sample.

    y_true : array-like of shape (n_samples,)

    Returns
    -------
    crps : float or array-like
        The CRPS value mean, or array of individual CRPS values.

    """
    # taken from https://github.com/WillyChap/ARML_Probabilistic/blob/main/Coastal_Points/Testing_and_Utility_Notebooks/CRPS_Verify.ipynb
    # and work by Louis Poulain--Auzeau (https://github.com/louisPoulain)
    reduction = kwargs.get("reduction", "mean")

    mu = y_pred[:, 0]
    sigma = y_pred[:, 1]

    # prevent negative sigmas
    sigma = torch.sqrt(sigma.pow(2))

    loc = (y_true - mu) / sigma
    pdf = torch.exp(-0.5 * loc.pow(2)) / torch.sqrt(
        2 * torch.from_numpy(np.array(np.pi))
    )
    cdf = 0.5 * (1.0 + torch.erf(loc / torch.sqrt(torch.tensor(2.0))))

    # compute CRPS for each (input, truth) pair
    crps = sigma * (
        loc * (2.0 * cdf - 1.0)
        + 2.0 * pdf
        - 1.0 / torch.from_numpy(np.array(np.sqrt(np.pi)))
    )

    return crps.mean() if reduction == "mean" else crps


def CRPSLoss(y_pred, y_true, **kwargs):
    reduction = kwargs.get("reduction", "mean")
    loss = 0.0
    indiv_losses = np.empty((y_true.shape))

    for i in range(0, y_true.shape[1]):
        l = CRPS_ML(y_pred[:, i * 2 : (i + 1) * 2], y_true[:, i], reduction=reduction)
        loss += l
        indiv_losses[:, i] = l.detach().cpu().numpy()

    return loss / (y_true.shape[1] // 2), indiv_losses


def CRPS_np(mu, sigma, y, **kwargs):
    reduction = kwargs.get("reduction", "mean")
    sigma = np.sqrt(sigma**2)
    loc = (y - mu) / sigma
    pdf = np.exp(-0.5 * loc**2) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1.0 + erf(loc / np.sqrt(2)))
    crps = sigma * (loc * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / np.sqrt(np.pi))
    return crps.mean() if reduction == "mean" else crps


def CRPSNumpy(mu_pred, sigma_pred, y_true, **kwargs):
    reduction = kwargs.get("reduction", "mean")
    indiv_losses = np.empty((y_true.shape))

    for i in range(0, y_true.shape[1]):
        l = CRPS_np(mu_pred[:, i], sigma_pred[:, i], y_true[:, i], reduction=reduction)
        indiv_losses[:, i] = l

    return indiv_losses


def summarize_performance(y_true, y_pred, y_baseline, metrics: list, **kwargs):
    """Summarize the performance of the model and the baseline.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The true target values.

    y_pred : array-like of shape (n_samples,)
        The predicted target values.

    y_baseline : array-like of shape (n_samples,)
        The baseline target values.

    metrics : list of functions
        The list of metrics to compute.

    Returns
    -------
    performance : dict
        A dictionary containing the performance metrics.
    """
    # # Check that the inputs are of the correct form
    # y_true, y_pred = _check_regression(y_true, y_pred)
    # y_true, y_baseline = _check_regression(y_true, y_baseline)

    # Assert that the predictions have the shape (n_samples, n_features)
    assert y_pred.ndim == 2, "y_pred must have shape (n_samples, n_features)"

    y_labels = kwargs.get("y_labels", {0: "Wind", 1: "Pressure"})

    # Compute the performance metrics
    performance = {}
    for metric in metrics:
        for i in range(y_pred.shape[1]):
            y_true_i = y_true[:, i]
            y_pred_i = y_pred[:, i]
            y_baseline_i = y_baseline[:, i]

            if metric.__name__ in short_names.keys():
                metric_name = short_names[metric.__name__]
            else:
                metric_name = metric.__name__

            performance[f"{metric_name}_{y_labels[i]}"] = metric(y_true_i, y_pred_i)
            performance[f"{metric_name}_{y_labels[i]}_baseline"] = metric(
                y_true_i, y_baseline_i
            )

            if kwargs.get("skill", True):
                performance[f"{metric_name}_{y_labels[i]}_skill"] = (
                    1
                    - performance[f"{metric_name}_{y_labels[i]}"]
                    / performance[f"{metric_name}_{y_labels[i]}_baseline"]
                )
        # performance[metric.__name__] = metric(y_true, y_pred)
        # performance[f"{metric.__name__}_baseline"] = metric(y_true, y_baseline)

        # if kwargs.get("skill", True):
        #     performance[f"{metric.__name__}_skill"] = 1 - (
        #         performance[metric.__name__]
        #         / performance[f"{metric.__name__}_baseline"]
        #     )

    return performance


# %%
def plot_performance(metrics: dict, ax, **kwargs):
    """Plot the performance metrics.

    Parameters
    ----------
    metrics : dict
        A dictionary containing the performance metrics.

    ax : matplotlib.axes.Axes
        The axes to plot the metrics on.

    Returns
    -------
    None
    """
    if kwargs.get("skill", True):
        assert isinstance(
            ax, np.ndarray
        ), "ax should be a numpy array if skill=True (default)"

        ax1 = ax[0]
        ax2 = ax[1]
    else:
        ax1 = ax

    model_name = kwargs.get("model_name", "Model")
    baseline_name = kwargs.get("baseline_name", "Baseline")

    # Generate a list of unique metrics
    metric_names = []
    for metric in metrics.keys():
        if not ("skill" in metric or "baseline" in metric):
            metric_names.append(metric)
    metric_names = np.unique(metric_names)

    # Define the colors for the bars
    colors = kwargs.get("colors", plt.cm.tab20.colors)

    ax1_labels = []
    # Plot the performance metrics
    for i, metric in enumerate(metric_names):
        model_metric = metrics[metric]
        baseline_metric = metrics[f"{metric}_baseline"]
        var = metric.split("_")[-1].lower()
        unit = units.get(var, "")
        ax1.bar(
            [i * 3],
            [model_metric],
            color=colors[i % len(colors)],
            label=f"{model_name} {metric} ({unit})",
            hatch=kwargs.get("model_hatch", None),
        )

        ax1.bar(
            [i * 3 + 1],
            [baseline_metric],
            color=colors[i % len(colors)],
            label=f"{baseline_name} {metric} ({unit})",
            hatch=kwargs.get("baseline_hatch", "//"),
        )

        ax1.set_ylabel("Score")
        ax1.set_title("Performance Metrics")

        ax1_labels += [f"{model_name} metric", f"{baseline_name} {metric}", ""]

        if kwargs.get("skill", True):
            ax2.bar(
                [i],
                [metrics[f"{metric}_skill"]],
                color=colors[i % len(colors)],
                label=f"{metric} Skill Score",
                hatch=kwargs.get("skill_hatch", None),
            )
            ax2.set_ylabel("Skill Score (1 - model/baseline)")
            ax2.set_title("Skill Scores")

    ax1.set_xticks(range(len(metric_names) * 3))
    ax1.set_xticklabels([""] * len(ax1_labels))
    ax1.legend(loc="lower right", framealpha=0.5)

    if kwargs.get("skill", True):
        ax2.set_xticks(range(len(metric_names)))
        ax2.set_xticklabels([""] * len(metric_names))
        ax2.legend(loc="lower right", framealpha=0.5)
