#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:10:00 2024

This file contains the data handling library - functions that will be
used in other scripts to perform the data tasks associated with TCBench

@author: mgomezd1
"""

# %% Imports
import dask.array as da
from dask_ml import preprocessing
import numpy as np


# %% Classes
## TODO: Add docstrings, add type hints, add error handling
## TODO: add logging
## TODO: implement other scaling methods (e.g., min-max)
class AI_StandardScaler(preprocessing.StandardScaler):
    # The fit function should expect dask arrays
    def fit(self, X, y=None):
        assert isinstance(X, da.Array), "X should be a dask array"
        self.mean_ = np.tile(
            X.mean(axis=(0, -1, -2)).compute().reshape(-1, 1, 1), (1, 241, 241)
        )
        self.scale_ = np.tile(
            X.std(axis=(0, -1, -2)).compute().reshape(-1, 1, 1), (1, 241, 241)
        )
        return self


# %%
