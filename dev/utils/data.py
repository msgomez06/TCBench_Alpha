#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:10:00 2024

This file contains the data handling library - functions that will be
used in other scripts to perform the data tasks associated with TCBench

@author: mgomezd1
"""
# %% Imports

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

default = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ECMWF/ERA5/"


# %% Classes
class Data_Collection:
    # The data collection class is a wrapper for xarray datasets that allows for
    # easier handling of folder structures and file naming conventions. It
    # also keeps track of what variables are available in the dataset and
    # the years that are available in the dataset.
    def __init__(
        self,
        data_path: str,  # Path to the data storage directory
        var_types: list = [  # Types of variables to load
            "SL",  # Surface / Single level
            "PL",  # Pressure Level
            "CV",  # Calculated Values, assumed single level
        ],
        file_type: str = "nc",  # File type of the data, netcdf by default
        **kwargs,
    ):
        assert os.path.isdir(
            data_path
        ), "The path to the data storage directory does not exist."

        # Save the keyword arguments as attributes
        for key, val in kwargs.items():
            setattr(self, key, val)

        self.data_path = data_path
        self.var_types = var_types
        self.file_type = file_type

        # Initialize the data collection object
        self._init_data_collection()

    def _init_data_collection(self):
        dir_contents = os.listdir(self.data_path)

        for var_type in self.var_types:
            assert (
                var_type in dir_contents
            ), f"The variable folder {var_type} is not present in the data storage directory."

            # Check what variables are available for each variable type
            self._check_vars(var_type)

    def _check_vars(self, var_type: str):
        # Check what variables are available for each variable type
        var_path = os.path.join(self.data_path, var_type)
        var_list = os.listdir(var_path)

        var_dictionary = {}
        global_year_list = []
        for var in var_list:
            file_list = sorted(os.listdir(os.path.join(var_path, var)))

            avail_years = []
            # Assert that all files are nc files
            for file in file_list:
                assert (
                    file.split(".")[-1] == self.file_type
                ), f"{file} is not a(n) {self.file_type} file"
                avail_years.append(file.split(".")[0].split("_")[1])
            global_year_list += avail_years
            var_dictionary[var] = sorted(avail_years)
        global_year_list = sorted(list(set(global_year_list)))

        if hasattr(self, "meta_dfs"):
            self.meta_dfs[var_type] = self._check_availability(
                var_dictionary, global_year_list
            )
        else:
            self.meta_dfs = {
                var_type: self._check_availability(var_dictionary, global_year_list)
            }

    def _check_availability(self, var_dict, all_years):
        # Initialize an empty DataFrame with the variable names as the index and the years as the columns
        df = pd.DataFrame(index=var_dict.keys(), columns=all_years)

        # Fill the DataFrame with booleans indicating whether each variable is available for each year
        for var, years in var_dict.items():
            for year in all_years:
                df.loc[var, year] = year in years

        # Convert the booleans to integers
        df = df.astype(int)

        return df

    # Function to print the availability of variables
    def variable_availability(self, **kwargs):
        assert hasattr(
            self, "meta_dfs"
        ), "The data collection object has not been properly initialized."

        save_image = kwargs.get("save_image", False)
        save_path = kwargs.get("save_path", None)

        assert (save_image and save_path) or (
            not save_image and not save_path
        ), "If save_image is True, save_path must be specified."

        for key in dc.meta_dfs.keys():
            # Create a colormap
            cmap = mcolors.ListedColormap(
                ["black"]
                + (list(plt.cm.tab20c.colors) * 10)[: len(dc.meta_dfs[key].index)]
            )
            norm = mcolors.Normalize(vmin=-0.5, vmax=len(dc.meta_dfs[key].index) + 0.5)
            fig, ax = plt.subplots()
            ax.matshow(
                dc.meta_dfs[key].to_numpy()
                * (np.arange(0, len(dc.meta_dfs[key].index)) + 1).reshape(-1, 1),
                cmap=cmap,
                norm=norm,
            )
            ax.set_yticks(np.arange(0, len(dc.meta_dfs[key].index)))
            ax.set_xticks(np.arange(0, len(dc.meta_dfs[key].columns)))
            ax.set_xticklabels(
                dc.meta_dfs[key].columns, rotation="vertical", ha="center"
            )
            ax.set_yticklabels(dc.meta_dfs[key].index)
            plt.show()

            if save_image:
                plt.savefig(save_path + key + ".png")


# %% Functions

# %% Test running the data collection class

if __name__ == "__main__":
    # Test running the data collection class
    dc = Data_Collection(default)

    # # Test the variable availability function
    dc.variable_availability()

    # for key in dc.meta_dfs.keys():
    #     # Create a colormap
    #     cmap = mcolors.ListedColormap(
    #         ["black"] + (list(plt.cm.tab20c.colors) * 10)[: len(dc.meta_dfs[key].index)]
    #     )
    #     norm = mcolors.Normalize(vmin=-0.5, vmax=len(dc.meta_dfs[key].index) + 0.5)
    #     fig, ax = plt.subplots()
    #     ax.matshow(
    #         dc.meta_dfs[key].to_numpy()
    #         * (np.arange(0, len(dc.meta_dfs[key].index)) + 1).reshape(-1, 1),
    #         cmap=cmap,
    #         norm=norm,
    #     )
    #     ax.set_yticks(np.arange(0, len(dc.meta_dfs[key].index)))
    #     ax.set_xticks(np.arange(0, len(dc.meta_dfs[key].columns)))
    #     ax.set_xticklabels(dc.meta_dfs[key].columns, rotation="vertical", ha="center")
    #     ax.set_yticklabels(dc.meta_dfs[key].index)
    #     plt.show()

# %%
