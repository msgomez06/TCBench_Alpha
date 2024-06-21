# %% Imports
# OS and IO
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dask.array as da
from dask.ml import preprocessing

# Backend Libraries
import joblib as jl

from utils import toolbox, ML_functions as mlf
from utils import data_lib as dlib
import metrics.metrics as metrics

# %%
datadir = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha"

# %% Data Loading
rng = np.random.default_rng(seed=42)

years = list(range(2013, 2020))
rng.shuffle(years)

sets, data = toolbox.get_sets(
    {"train": years[:-2], "test": [2020], "validation": years[-2:]},
    datadir=datadir,
    test_strategy="custom",
    # use_cached=False,
    # verbose=True,
    # debug=True,
)

# %% Reload for interactive development
try:
    import importlib

    importlib.reload(TCBench.baselines.baselines)
    importlib.reload(metrics)
except:
    import tcbenchmark.baselines.baselines as baselines

# %% MLR Training

# We begin by instantiating our baseline model
MLR = baselines.TC_DeltaIntensity_MLR()

# And fitting it using the training data. Note that we don't
# normalize the inputs
MLR.fit(
    base_intensity=data["train"]["base_intensity"],
    AI_X=data["train"]["inputs"],
    y=data["train"]["outputs"],
)
# %% MLR Validation

# We then use the model to predict the validation data
y_hat = MLR.predict(
    base_intensity=data["validation"]["base_intensity"],
    AI_X=data["validation"]["inputs"],
)

# %% Baseline

# We want to compare the model to a simple baseline, in this case
# the persistence model. Note that this is simply the base intensity
# we're predicting the delta for.
y_persistence = data["validation"]["base_intensity"]

# %% Evaluation

# Let's start by loading the validation outputs into a variable
# for easier access
y_true = data["validation"]["outputs"]

# %% Compute y_true, y_hat, y_persistence
y_true = y_true.compute()
y_hat = y_hat.compute()
y_persistence = y_persistence.compute()

# %% Evaluation
# We will want to evaluate the model using the Root Mean Squared Error and
# the Mean Absolute Error, as well as the associated skill scores compared
# to the persistence model

# Importing the metrics
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

global_performance = metrics.summarize_performance(
    y_true,  # Ground truth
    y_hat,  # Model prediction
    y_persistence,  # Baseline, used for skill score calculations
    [root_mean_squared_error, mean_absolute_error],
)
# %%
# Plotting Global Performance
fig, axes = plt.subplots(
    1, 2, figsize=(15, 5), dpi=150, gridspec_kw={"width_ratios": [2, 1]}
)
metrics.plot_performance(
    global_performance, axes, model_name="MLR", baseline_name="Persistence"
)
toolbox.plot_facecolors(fig=fig, axes=axes)
plt.show()
plt.close()

# %%
# Let's also go ahead and do the analysis per lead time
unique_leads = np.unique(data["validation"]["leadtime"].compute())
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
    lead_mask = data["validation"]["leadtime"] == lead

    y_true_lead = y_true[lead_mask]
    y_hat_lead = y_hat[lead_mask]
    y_persistence_lead = y_persistence[lead_mask]

    lead_performance = metrics.summarize_performance(
        y_true_lead,  # Ground truth
        y_hat_lead,  # Model prediction
        y_persistence_lead,  # Baseline, used for skill score calculations
        [root_mean_squared_error, mean_absolute_error],
    )

    metrics.plot_performance(
        lead_performance, axes[idx], model_name="MLR", baseline_name="Persistence"
    )
    toolbox.plot_facecolors(fig=fig, axes=axes[idx])

    # Append the lead time to the title for both axes
    axes[idx][0].set_title(f"\n Lead Time: +{lead}h \n" + axes[idx][0].get_title())
    axes[idx][1].set_title(f"\n Lead Time: +{lead}h \n" + axes[idx][1].get_title())

# %%
