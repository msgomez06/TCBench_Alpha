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
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import warnings
import xarray as xr
from dask import optimize

# %% Classes
## TODO: Add docstrings, add type hints, add error handling
## TODO: add logging
## TODO: implement other scaling methods (e.g., min-max)

## TODO: Properly parallelize the standard scaler
class AI_StandardScaler(preprocessing.StandardScaler):
    # The fit function should expect dask arrays
    def fit(self, X, y=None, **kwargs):
        assert isinstance(X, da.Array), "X should be a dask array"
        num_workers = kwargs.get("num_workers", 1)
        mean = X.mean(axis=(0, -1, -2)).compute(num_workers=num_workers, scheduler="threads")
        std = X.std(axis=(0, -1, -2)).compute(num_workers=num_workers, scheduler="threads")
        self.mean_ = np.tile(
            mean.reshape(-1, 1, 1), (1, 241, 241)
        )
        self.scale_ = np.tile(
            std.reshape(-1, 1, 1), (1, 241, 241)
        )
        return self


class DaskDataset(Dataset):
    def __init__(
            self, 
            AI_X, 
            AI_scaler, 
            base_int, 
            base_scaler,
            target_data, 
            **kwargs):
        # Assert that AI_X and base_int are dask arrays
        assert isinstance(AI_X, da.Array), "AI_X should be a dask array"
        # and that they have the same length
        assert (
            AI_X.shape[0] == base_int.shape[0]
        ), "AI_X and base_int should have the same length"

        self.AI_X = AI_X
        self.AI_scaler = AI_scaler
        self.base_int = base_int
        self.base_scaler = base_scaler
        self.target_data = target_data
        self.set_name = kwargs.get("set_name", "unnamed")

        self.chunk_size = kwargs.get("chunk_size", 32)

        # # Attempt at optimizing dask array
        # self.AI_X = optimize(self.AI_X.rechunk((self.chunk_size, 5, 241, 241)))[0]
        # self.base_int = optimize(self.base_int.rechunk((self.chunk_size, 2)))[0]

        # Attempt at optimizing dask array
        # self.AI_X = self.AI_X.rechunk((self.chunk_size, 5, 241, 241))
        # self.base_int = self.base_int.rechunk((self.chunk_size, 2))

        # Scale the data using the scaler using dask.delayed.ravel

        interim = self.AI_X.rechunk((self.chunk_size, 5, 241, 241)).to_delayed().ravel()
        interim = [da.from_delayed(self.AI_scaler.transform(block), shape=(min(self.chunk_size, self.AI_X.shape[0] - i*self.chunk_size), 5, 241, 241), dtype=self.AI_X.dtype) for i, block in enumerate(interim)]
        self.AI_X = da.concatenate(interim)

        # self.AI_X = da.map_blocks(
        #     self.AI_scaler.transform, self.AI_X, dtype=np.float32, chunks=self.AI_X.chunksize)
        # self.base_int = da.map_blocks(
        #     self.base_scaler.transform, self.base_int, dtype=np.float32, chunks=self.base_int.chunksize)

        if kwargs.get("load_into_memory", False):
            self.AI_X = self.AI_X.compute()
            self.base_int = self.base_int.compute()
            self.target_data = self.target_data.compute()
        else:

            zarr_name = kwargs.get("zarr_name", "unnamed")
            cachedir = kwargs.get("cachedir", os.path.join(os.getcwd(), 'cache'))
            zarr_path = os.path.join(cachedir, f"{zarr_name}.zarr")

            if not os.path.exists(zarr_path):
                os.makedirs(zarr_path)

            # save AI_X to zarr
            AI_X_path = os.path.join(zarr_path, "AI_X")
            if not os.path.exists(AI_X_path):
                self.AI_X.rechunk((self.chunk_size, 5, 241, 241)).to_zarr(AI_X_path)
                # self.AI_X.to_zarr(AI_X_path)
            elif kwargs.get("overwrite", True):
                # overwrite
                self.AI_X.to_zarr(AI_X_path, overwrite=True)
            
            # load AI_X from zarr
            self.AI_X = da.from_zarr(AI_X_path)

        self.device = kwargs.get("device", torch.device('cuda') if torch.cuda.is_available else torch.device("cpu"))

    def __len__(self):
        return len(self.base_int)

    def __getitem__(self, idx):
        # # Load the specific data points needed for this index
        # # Convert them to PyTorch tensors
        # AI_sample = torch.tensor(self.AI_X[idx].compute(), dtype=torch.float32)
        # base_int_sample = torch.tensor(self.base_int[idx], dtype=torch.float32)
        # target_sample = torch.tensor(self.target_data[idx], dtype=torch.float32)

        # return AI_sample, base_int_sample, target_sample
        
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")  # Ignore all warnings
        AI_sample = self.AI_X[idx]
        base_int_sample = self.base_int[idx]
        target_sample = self.target_data[idx]
        if base_int_sample.ndim == 1:
            base_int_sample = base_int_sample[None, :]

        sample = (AI_sample, base_int_sample, target_sample)

        # sample = self.data_processor(AI_sample, base_int_sample)
        # sample = (*sample, target_sample)
        output = []
        for array in sample:
            if isinstance(array, np.ndarray):
                array = torch.from_numpy(array.astype(np.float32)).to(
                    self.device
                )
                output.append(array)
            elif isinstance(array, da.Array):
                array = torch.from_numpy(array.compute().astype(np.float32)).to(
                    self.device
                )
                output.append(array)
        return (*output,)


# %% Functions
def make_dataloader(dataset, **kwargs):
    dataset
    return DataLoader(dataset, **kwargs)
