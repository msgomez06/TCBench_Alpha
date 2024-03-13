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
import matplotlib
import xarray as xr
import metpy.calc as mpcalc
import metpy.units as mpunits
import joblib as jl
from memory_profiler import profile
from pint import Quantity

# TCBench Libraries
try:
    from utils import constants
except:
    import constants

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
            # Check if var is a directory. If not, pass and continue
            # but print a warning
            if not os.path.isdir(os.path.join(var_path, var)):
                print(f"{var} is not a directory. Skipping.")
                continue

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
        if global_year_list == []:
            print("No files found in " + var_path)
        else:
            global_year_list = [int(yr) for yr in set(global_year_list)]
            global_year_list = np.arange(
                min(global_year_list), max(global_year_list) + 1
            ).astype(str)

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
        df.sort_index(inplace=True)
        return df

    # Function to print the availability of variables
    def variable_availability(self, **kwargs):
        assert hasattr(
            self, "meta_dfs"
        ), "The data collection object has not been properly initialized."

        save_path = kwargs.get("save_path", None)

        assert (save_path is None) or (
            os.path.isdir(save_path)
        ), "Invalid image save_path - make sure the path exists."

        matplotlib.rc(
            "xtick", labelsize=kwargs.get("tick_label_size", 6)
        )  # fontsize of the tick labels
        matplotlib.rc(
            "ytick", labelsize=kwargs.get("tick_label_size", 6)
        )  # fontsize of the tick labels

        for key in dc.meta_dfs.keys():
            # Create a colormap
            cmap = mcolors.ListedColormap(
                ["black"]
                + (list(plt.cm.tab20c.colors) * 10)[: len(dc.meta_dfs[key].index)]
            )
            norm = mcolors.Normalize(vmin=-0.5, vmax=len(dc.meta_dfs[key].index) + 0.5)

            # Create the figure
            fig, ax = plt.subplots(
                dpi=kwargs.get("dpi", 300),
            )

            # Set the title using the key
            fig.suptitle(
                f"Variable Availability for {constants.data_store_names.get(key, key)}"
            )
            # Plot the availability matrix
            ax.imshow(
                dc.meta_dfs[key].to_numpy()
                * (np.arange(0, len(dc.meta_dfs[key].index)) + 1).reshape(-1, 1),
                cmap=cmap,
                norm=norm,
            )

            # Set aspect ratio according to the number of variables and years
            aspect_ratio = (
                len(dc.meta_dfs[key].index) / len(dc.meta_dfs[key].columns)
                if dc.meta_dfs[key].columns.any() and dc.meta_dfs[key].index.any()
                else 1
            )
            ax.set_box_aspect(aspect_ratio)

            # Set the tick labels
            ax.set_yticks(np.arange(0, len(dc.meta_dfs[key].index)))
            ax.set_xticks(np.arange(0, len(dc.meta_dfs[key].columns)))
            ax.set_xticklabels(
                dc.meta_dfs[key].columns, rotation="vertical", ha="center"
            )
            ax.set_yticklabels(dc.meta_dfs[key].index)

            # tight layout
            fig.tight_layout()
            fig.subplots_adjust(left=0.35)

            if save_path is not None:
                fig.savefig(save_path + key + ".png")

            plt.show()

    def retrieve_ds(self, vars: list, years: list, **kwargs):
        assert hasattr(
            self, "meta_dfs"
        ), "The data collection object has not been properly initialized."

        # check that the vars are a list or a string
        assert isinstance(vars, list) or isinstance(
            vars, str
        ), "vars must be a list or a string"

        # check if the variables are a list, else make it a list
        if not isinstance(vars, list):
            vars = [vars]

        # check that the years are an int or a list
        assert isinstance(years, int) or isinstance(
            years, list
        ), "years must be an int or a list"

        # check if the years are a list, else make it a list
        if not isinstance(years, list):
            years = [years]

        # Initialize list of files for multi-file dataset
        file_list = []

        data_var_dict = {}
        # Check that the variables are available
        for var in vars:
            # Assert that the variable is a string
            assert isinstance(
                var, str
            ), f"Variable {var} is not a string. Aborting data load operation."

            # Check that the variable is available in one of the groups. If not available
            # or available in more than one group, abort the data load operation
            group = None
            for key in self.meta_dfs.keys():
                if var in self.meta_dfs[key].index:
                    if group is None:
                        group = key
                    else:
                        raise ValueError(
                            f"{var} is available in more than one group. Aborting data load operation."
                        )
            assert (
                group is not None
            ), f"{var} is not available in any of the groups. Aborting data load operation."

            # and check that the years are available
            for year in years:
                # Assert that the year is an integer
                assert isinstance(
                    year, int
                ), f"Year {year} is not an integer. Aborting data load operation."

                assert (
                    self.meta_dfs[group].loc[var].loc[str(year)] == 1
                ), f"{year} is not available for {var}. Aborting data load operation."

                file_list.append(
                    os.path.join(
                        self.data_path,
                        group,
                        var,
                        f"{kwargs.get('prefix', 'ERA5')}_{year}_{var}.{kwargs.get('file_type', 'nc')}",
                    )
                )
                if var not in data_var_dict.keys():
                    data_var_dict[var] = list(xr.open_dataset(file_list[-1]).data_vars)[
                        0
                    ]

        # Load the dataset
        print(f"Opening file list: {file_list}")
        ds = xr.open_dataset(file_list[0])
        # ds = kwargs.get("data_loader", xr.open_mfdataset)(
        #     file_list,
        #     #**kwargs.get("data_loader_kwargs", {"parallel": True}),
        # )

        return ds, data_var_dict

    # TODO: Figure out why this explodes memory usage and fix it
    @profile
    def calculate_field(
        self,
        function: callable,
        argument_names: dict,
        years,
        **kwargs,
    ):
        """
        This function calculates a field using the given function and
        the given variables. It calculates and saves the field for the
        requested years. If the field is already available the function
        will not calculate the field again.

        Parameters
        ----------
        function : callable
            The function used to calculate the value
        argument_names : dict
            The argument names for the function, with the variable names as keys
            and the argument names as values
        years : list

        Returns
        -------
        None or xarray.Dataset
        """
        # Check if the function is a metpy callable
        if hasattr(mpcalc, function.__name__):
            # if it is, check that the required units are passed as kwargs
            assert (
                "units" in kwargs.keys()
            ), f"Units are required for {function.__name__} since it's a metpy function."

        # check that the argument names are a dict
        assert isinstance(
            argument_names, dict
        ), "argument_names must be a dict mapping variable names to argument names"

        # check that the years are an int or a list. Then, if it's an int, make it a list
        assert isinstance(years, int) or isinstance(
            years, list
        ), "years must be an int or a list"
        if not isinstance(years, list):
            years = [years]

        # get list of variables
        var_list = list(argument_names.keys())

        pressure_bool = "pressure" in var_list

        # check if pressure is in the variable names
        if pressure_bool:
            print(
                "Pressure detected in the argument names. "
                "Since pressure is a coordinate variable, it will be "
                "taken from the dataset coordinates."
            )
            var_list.remove("pressure")

        # Check if the save directory exists, if not create it
        save_dir = os.path.join(
            kwargs.get("save_path", self.data_path),
            kwargs.get("save_folder", "CV"),
            kwargs.get("save_var", function.__name__),
        )
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            avail_list = None
        else:
            # Check what years are already available
            avail_list = os.listdir(save_dir)
            avail_years = []
            for file in avail_list:
                avail_years.append(file.split(".")[0].split("_")[1])
            avail_years = [int(yr) for yr in sorted(list(set(avail_years)))]
            print(
                f"Calculated files for {function.__name__} detected for years {avail_years}"
            )
            if len(avail_list) < 1:
                avail_list = None

        # load the dataset
        ds, dv_dict = self.retrieve_ds(var_list, years, **kwargs)

        # check if specific levels have been requested
        if "levels" in kwargs.keys():
            print("Levels detected in the kwargs. " "Filtering to requested levels.")
            # print("Levels existing in any of the available files will be skipped!")
            # if avail_list is not None:
            #     # determine what levels are available
            #     calculated_levels = xr.open_mfdataset(
            #         [os.path.join(save_dir, file) for file in avail_list]
            #     ).level.values

            #     for level in kwargs.get("levels"):
            #         if level in calculated_levels:
            #             print(f"Level {level} already calculated. Skipping.")
            #             kwargs.get("levels").remove(level)

            ds = ds.sel(level=ds.level.isin(kwargs.get("levels"))).load()

        print("Dataset loaded. Calculating field...")
        arg_dict = {}
        for var in argument_names.keys():
            arg_dict[argument_names[var]] = (
                ds[dv_dict[var]] * mpunits.units(kwargs.get("units", {}).get(var, None))
                if var != "pressure"
                else ds[kwargs.get("pressure_name", "level")]
                * mpunits.units(kwargs.get("units", {}).get("pressure", None))
            )

        # Generate the function arguments for the offset and scale factor
        offset_dict = {}
        for var in argument_names.keys():
            offset_dict[argument_names[var]] = (
                ds[dv_dict[var]].encoding["add_offset"]
                * mpunits.units(kwargs.get("units", {}).get(var, None))
                if var != "pressure"
                else ds[kwargs.get("pressure_name", "level")]
                * mpunits.units(kwargs.get("units", {}).get("pressure", None))
            )

        scale_dict = {}
        for var in argument_names.keys():
            scale_dict[argument_names[var]] = (
                ds[dv_dict[var]].encoding["scale_factor"]
                * mpunits.units(kwargs.get("units", {}).get(var, None))
                if var != "pressure"
                else ds[kwargs.get("pressure_name", "level")]
                * mpunits.units(kwargs.get("units", {}).get("pressure", None))
            )

        for key in scale_dict.keys():
            if key != "pressure":
                scale_dict[key] = scale_dict[key] + offset_dict[key]

        # Calculate the value for the offset and scale factor from the input data
        offset = function(**offset_dict)
        if isinstance(offset, Quantity):
            offset = offset.magnitude
        print("Offset calculated: ", offset)

        scale = function(**scale_dict)
        if isinstance(scale, Quantity):
            scale = scale.magnitude
        scale -= offset
        print("Scale factor calculated: ", scale)

        # Set it up so that the function is applied to each time step using joblib
        def time_process(function, step, arg_dict):
            # calculate the field
            temp_dict = {}
            for arg in arg_dict.keys():
                if arg != "pressure":
                    temp_dict[arg] = arg_dict[arg].sel(time=step)
                else:
                    temp_dict[arg] = arg_dict[arg]
                print(temp_dict[arg])

            # calculate the field
            data = function(**temp_dict)
            units = kwargs.get("out_units", str(data.metpy.units))
            print("Fine till here")
            # create a dataset from the dataarray
            data = (
                data.metpy.dequantify()
                .to_dataset(
                    name=function.__name__,
                )
                .compute()
            )

            # add the attributes
            data.attrs = {
                "units": units,
                "long_name": kwargs.get("long_name", function.__name__),
                "standard_name": kwargs.get("standard_name", function.__name__),
                "Calculation Source": 'Calculated using the TCBench "calculate_field" function',
            }
            print("Fine till save")
            if kwargs.get("save", False):
                print("Saving temporary file...")
                # compression code inspired by github.com/pydata/xarray/discussions/5709
                # and https://stackoverflow.com/questions/70102997/scale-factor-and-add-offset-in-xarray-to-netcdf-lead-to-some-small-negative-valu
                # if you run into issues, make sure netcdf4 is installed in conda env
                # (netcdf4 solution from queez in https://stackoverflow.com/questions/40766037/)
                encoding = {}
                for data_var in data.data_vars:
                    encoding[data_var] = {
                        "original_shape": data[data_var].shape,
                        "_FillValue": kwargs.get("fill_value", -32767),
                        "dtype": kwargs.get("save_dtype", np.int16),
                        "add_offset": offset,
                        "scale_factor": scale,
                    }

                print(data.time.dt.year.values)

                target_file = os.path.join(
                    save_dir,
                    kwargs.get(
                        "save_name",
                        f"temp_{step}_"
                        f"{kwargs.get('prefix', 'ERA5')}_"
                        f"{str(data['time.year'].values)[:4]}_{kwargs.get('save_var', function.__name__)}"
                        f".{kwargs.get('file_type', 'nc')}",
                    ),
                )

                data.to_netcdf(
                    target_file,
                    mode="w",
                    encoding=encoding,
                    engine=kwargs.get("engine", "netcdf4"),
                    # compute=True,
                )
                data.close()
            # else:
            #     return data

        # use joblib to parallelize the calculation per time step
        # with as many threads as there are cores
        job_array = jl.Parallel(
            n_jobs=-1,
            verbose=2,
        )(jl.delayed(time_process)(function, i, arg_dict) for i in ds.time.values)

        if kwargs.get("save", False):
            print("Merging temporary files...")
            # make the list of temporary files
            for year in years:
                temp_files = os.listdir(save_dir)
                # filter to make sure the files are from the year being processed
                temp_files = [
                    file
                    for file in temp_files
                    if file.split(".")[0].split("_")[1] == str(year)
                ]
                # check if a non temporary file is in the list
                if len(temp_files) > 0:
                    non_temp_files = [
                        file
                        for file in temp_files
                        if file.split(".")[0].split("_")[0] != "temp"
                    ]
                    if len(non_temp_files) == 1:
                        # TODO: handling the merging of an existing file
                        pass
                    elif len(non_temp_files) > 1:
                        raise ValueError("Multiple base files found - aborting.")

                # check to make sure the remaining files are temporary files
                temp_files = [
                    file
                    for file in temp_files
                    if file.split(".")[0].split("_")[0] == "temp"
                ]
                xr.open_mfdataset(
                    [os.path.join(save_dir, file) for file in temp_files]
                ).to_netcdf(
                    os.path.join(
                        save_dir,
                        f"{kwargs.get('prefix', 'ERA5')}_{year}_{kwargs.get('save_var', function.__name__)}.nc",
                    ),
                    mode="w",
                    encoding={
                        "add_offset": offset,
                        "scale_factor": scale,
                        "_FillValue": kwargs.get("fill_value", -32767),
                        "dtype": kwargs.get("save_dtype", np.int16),
                    },
                    engine=kwargs.get("engine", "netcdf4"),
                )

                # remove the temporary files
                for file in temp_files:
                    os.remove(os.path.join(save_dir, file))

        else:
            return xr.concat(job_array)


# %% Functions

# %% Test running the data collection class
if __name__ == "__main__":
    # Test running the data collection class
    dc = Data_Collection(default)

    # Test the variable availability function
    dc.variable_availability(
        # save_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ECMWF/ERA5/"
    )

    print(dc.meta_dfs)

    ## Test the retrieve_ds function
    # ds = dc.retrieve_ds(
    #         [
    #             "10m_u_component_of_wind",
    #             "mean_sea_level_pressure",
    #             "u_component_of_wind",
    #         ],
    #         1999,
    #     )

    # # Test the calculate_field function
    # dc.calculate_field(
    #     function=mpcalc.potential_temperature,
    #     argument_names={"temperature": "temperature", "pressure": "pressure"},
    #     units={"temperature": "kelvin", "pressure": "millibar"},
    #     # chunking={"chunks": {"time": 200}},
    #     levels=950,
    #     years=2020,
    #     save=True,
    # ).chunk({"time": 200}).to_netcdf("/scratch/mgomezd1/test.nc")

# %%
