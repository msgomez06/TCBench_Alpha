#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:45:42 2023

This file contains a "toolbox" library - functions and constants that will be
used in other scripts to perform the tasks associated with TCBench

@author: mgomezd1
"""

# %% Imports and base paths

# OS and IO
import os
import traceback
import subprocess

# Base Libraries
import pandas as pd
import numpy as np
import xarray as xr
import xesmf as xe

# TCBench Libraries
from utils import constants

# Retrieve Repository Path
repo_path = "/" + os.path.join(*os.getcwd().split("/")[:-1])

print(repo_path)


# %% Auxilliary Functions
def axis_generator(**kwargs):
    origin = kwargs.get("origin", (0, 0))
    resolution = kwargs.get("resolution", 0.25)
    lat_limits = kwargs.get("lat_limits", None)
    lon_limits = kwargs.get("lon_limits", None)
    lon_mode = kwargs.get("lon_mode", 360)
    use_poles = kwargs.get("use_poles", True)

    if lat_limits is not None:
        assert (
            origin[0] >= lat_limits[0]
        ), "Origin lower limit greater than origin value"
        assert origin[0] <= lat_limits[1], "Origin upper limit less than origin value"
    else:
        lat_limits = (-90, 90)

    if lon_limits is not None:
        assert (
            origin[1] >= lon_limits[0]
        ), "Origin lower limit greater than origin value"
        assert origin[1] <= lon_limits[1], "Origin upper limit less than origin value"
    else:
        lon_limits = (0, 360) if lon_mode == 360 else (-180, 180)

    lat_vector = np.hstack(
        [
            np.flip(
                np.arange(
                    origin[0], lat_limits[0] - resolution * use_poles, -resolution
                )
            ),
            np.arange(origin[0], lat_limits[1] + resolution * use_poles, resolution),
        ]
    )

    if lon_mode == 360:
        lon_vector = np.hstack(
            [
                np.flip(np.arange(origin[1], lon_limits[0] - resolution, -resolution)),
                np.arange(origin[1], lon_limits[1], resolution),
            ]
        )

    else:
        lon_vector = np.hstack(
            [
                np.flip(np.arange(origin[1], lon_limits[0], -resolution)),
                np.arange(origin[1], lon_limits[1] + resolution, resolution),
            ]
        )

    # Sanitize outputs
    lat_vector = np.unique(lat_vector)
    lon_vector = np.unique(lon_vector)

    return lat_vector, lon_vector


# Lat-lon grid generator
def ll_gridder(**kwargs):
    """
    ll_gridder is a lat-lon grid generator that's useful for generator the
    data masks associated with the

    Parameters
    ----------
    origin : TYPE, optional
        DESCRIPTION. The default is (0,0).
    resolution : TYPE, optional
        DESCRIPTION. The default is 0.25.
    lon_mode : TYPE, optional
        DESCRIPTION. The default is 'pos'.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    origin = kwargs.get("origin", (0, 0))
    resolution = kwargs.get("resolution", 0.25)
    lon_mode = kwargs.get("lon_mode", 360)
    lat_limits = kwargs.get("lat_limits", None)
    lon_limits = kwargs.get("lon_limits", None)
    use_poles = kwargs.get("use_poles", True)

    assert isinstance(
        use_poles, bool
    ), f"Unexpected type for use_poles:{type(use_poles)}. Expected boolean."
    assert (
        lon_mode == 180 or lon_mode == 360
    ), f"Invalid long mode in ll_gridder: {lon_mode}. Expected 180 or 360 as ints"

    lat_vector, lon_vector = axis_generator(
        origin=origin,
        resolution=resolution,
        lat_limits=lat_limits,
        lon_limits=lon_limits,
        lon_mode=lon_mode,
        use_poles=use_poles,
    )

    lats, lons = np.meshgrid(lat_vector, lon_vector)

    return lats, lons


# Great circle distance calculations
def haversine(latp, lonp, lat_list, lon_list, **kwargs):
    """──────────────────────────────────────────────────────────────────────────┐
      Haversine formula for calculating distance between target point (latp,
      lonp) and series of points (lat_list, lon_list). This function can handle
      2D lat/lon lists, but has been used with flattened data

      Based on:
      https://medium.com/@petehouston/calculate-distance-of-two-locations-on-earth-using-python-1501b1944d97


      Inputs:
          latp - latitude of target point

          lonp - longitude of target point

          lat_list - list of latitudess (lat_p-1, lat_p-2 ... lon_p-n)

          lon_list - list of longitudess (lon_p-1, lon_p-2 ... lon_p-n)

      Outputs:

    └──────────────────────────────────────────────────────────────────────────"""
    kwargs.get("epsilon", 1e-6)

    latp = np.radians(latp)
    lonp = np.radians(lonp)
    lat_list = np.radians(lat_list)
    lon_list = np.radians(lon_list)

    dlon = lonp - lon_list
    dlat = latp - lat_list
    a = np.power(np.sin(dlat / 2), 2) + np.cos(lat_list) * np.cos(latp) * np.power(
        np.sin(dlon / 2), 2
    )

    # Assert that sqrt(a) is within machine precision of 1
    # assert np.all(np.sqrt(a) <= 1 + epsilon), 'Invalid argument for arcsin'

    # Check if square root of a is a valid argument for arcsin within machine precision
    # If not, set to 1 or -1 depending on sign of a
    a = np.where(np.sqrt(a) <= 1, a, np.sign(a))

    return 2 * 6371 * np.arcsin(np.sqrt(a))


# %% Data Processing / Preprocessing Library
"""
This cell contains the following classes and functions:
    
"""


def read_hist_track_file(
    tracks_path=f"{repo_path}/tracks/ibtracs/",
    backend=pd.read_csv,
    track_cols=constants.ibtracs_cols,
    skip_rows=[1],
):
    """
    Function to read storm track files that include all the historical data
    in a single file

    Parameters
    ----------
    tracks_path : STR, required
        Path to the folder holding the storm track in a single file.
        The default is f'{repo_path}/tracks/ibtracs'. By default the file
        is expected to be in csv format

    backend : FUNC, required
        file handler for loading the track file. The default is pandas'
        read_csv function.

    track_cols : track_cols object defined in constants.py, required
        Track columns object which contains data about the columns in the
        tracks file. Needs to at least be able to return the column names,
        column dtypes (excluding datetime types), and columns that should
        be parsed as dates.

    skip_rows : list, optional
        rows to skip when reading in data with pandas.read_csv
        Required when working with ibtracs, as there is a unit row after
        the header row, hence the default [1] value


    Returns
    -------
    track_object : type defined by backend. Default is pandas dataframe

    """
    file_list = os.listdir(tracks_path)

    assert len(file_list) == 1, f"{tracks_path} has more than one file. Aborting."

    with open(f"{tracks_path}/{file_list[0]}") as handle:
        if backend == pd.read_csv:
            data = backend(
                handle,
                usecols=track_cols.get_colnames(),
                dtype=track_cols.get_dtypes(),
                skiprows=skip_rows,
                parse_dates=track_cols.get_datetime_cols(),
                na_filter=False,  # Otherwise pandas interprets 'NA' as NaN
            )
        elif backend == "meteo_france":
            data = pd.read_csv(
                handle,
                usecols=track_cols.get_colnames(),
                dtype=track_cols.get_dtypes(),
                # parse_dates = {'date':[*track_cols.get_datetime_cols()]},
                sep=";",
            )
            datetime_cols = track_cols.get_datetime_cols()
            data["date"] = pd.to_datetime(
                data[datetime_cols]
                .copy()
                .rename(
                    columns={
                        "ANNEE": "year",
                        "MOIS": "month",
                        "JOUR": "day",
                        "HEURE_UTC": "hour",
                    }
                )
            )
        else:
            data = backend(handle)

    return data


def read_py_track_file(
    tracks_path=f"{repo_path}/tracks/sgstracks/", year=2000, backend=pd.read_csv
):
    """
    Function to read per-year separated storm track files

    Parameters
    ----------
    tracks_path : STR, required
        Path to the folder holding the storm tracks separated by year.
        The default is f'{repo_path}/tracks/'. By default the files are
        expected to be in csv format
    year : INT, required
        year of the tracks to load. The default is 2000.
    backend : FUNC, required
        file handler for loading the track file. The default is pandas'
        read_csv function.

    Returns
    -------
    track_object : type defined by backend. Default is pandas dataframe

    """
    # Retrieve the list of files from the path
    file_list = os.listdir(tracks_path)

    for idx, file in enumerate(file_list.copy()):
        if str(year) not in file:
            file_list.remove(file)

    assert (
        len(file_list) == 1
    ), f"{tracks_path} has more than one file for year {year}. Aborting."

    with open(f"{tracks_path}/{file_list[0]}") as handle:
        data = backend(handle)

    return data


def process_py_track_data(data, track_cols):
    for column in track_cols.__dict__.keys():
        assert column in data.columns, f"{column} not found in header columns"

        if isinstance(track_cols.__dict__[column], dict):
            print(column)

    # TODO:

    pass


# Functions to sanitize timestamp data
def sanitize_timestamps(timestamps, data):
    # Sanitize timestamps because ibtracs includes unsual time steps,
    # e.g. 603781 (Katrina, 2005) includes 2005-08-25 22:30:00,
    # 2005-08-29 11:10:00, 2005-08-29 14:45:00
    valid_steps = timestamps[np.isin(timestamps, data.time.values)]

    data_steps = data.sel(time=valid_steps)
    return data_steps


# %%
# Class used to process track file and instantiate a track per event


# Class presenting tracks
class tc_track:
    def __init__(self, UID, NAME, track, timestamps, **kwargs):
        try:
            assert isinstance(
                UID, str
            ), f"Unique Identifier Error. {type(UID)} given when string was expected"
            assert isinstance(
                NAME, str
            ), f"Name Error. {type(NAME)} given when string was expected"
            assert isinstance(
                track, np.ndarray
            ), f"Track Error. {type(track)} given when numpy array was expected"
            assert isinstance(timestamps, np.ndarray) or isinstance(
                timestamps, pd.DataFrame
            ), f"Unsupported timestamp iterator {type(timestamps)}"
            assert np.issubdtype(
                timestamps.dtype, np.datetime64
            ), f"Unsupported timestamp type: {timestamps.dtype} - expected datetime64"

            self.uid = UID
            self.name = NAME
            self.track = (track,)
            self.timestamps = timestamps

            # Add alternate ID if available
            self.ALT_ID = kwargs.get("ALT_ID", None)

            # Check to see if file exists for UID, if not, create it
            self.filepath = kwargs.get("filepath", repo_path + "/data/")

            if not os.path.exists(self.filepath):
                print("Filepath for processed data does not exist. Creating...")
                os.makedirs(self.filepath)
            else:
                # Get the filelist at the filepath
                file_list = os.listdir(self.filepath)

                # Filter out files that do not include the UID
                for idx, file in enumerate(file_list.copy()):
                    if self.uid not in file:
                        file_list.remove(file)

                # For each type of data, load the data into the object
                # Files follow <SID>.<TYPE>.<(S)ingle or (M)ultilevel>.nc naming convention
                for file in file_list:
                    # Get the variable name from the file name
                    ds_type = file.split(".")[1]
                    level_type = file.split(".")[2]
                    # Load the data into the object
                    setattr(
                        self,
                        ds_type + level_type + "ds",
                        xr.open_dataset(self.filepath + file),
                    )

        except Exception as e:
            print(
                f"Encountered and exception when instiating a tc_track object: \n{e}\n"
            )
            print(traceback.format_exc())

    def add_trackdata(
        self,
        var_dict: dict,
    ):
        """
        Function to add additional track data to the track object

        Parameters
        ----------
        var_dict : dict, required
            Dictionary containing the variable name as the key and the
            variable data as the value
        """
        for var, data in var_dict.items():
            self.__setattr__(var, data)

    def get_radmask(self, point, **kwargs):
        # read in parameters if submitted, otherwise use defaults
        radius = kwargs.get("radius", 500)
        grid = kwargs.get("grid", ll_gridder(**kwargs))
        distance_calculator = kwargs.get("distance_calculator", haversine)

        return (
            distance_calculator(
                point[0],
                point[1],
                grid[0],
                grid[1],
            )
            <= radius
        ).T[np.newaxis, :, :]

    def get_rectmask(self, point, **kwargs):
        # read in parameters if submitted, otherwise use defaults
        grid = kwargs.get("grid", ll_gridder(**kwargs))
        distance_calculator = kwargs.get("distance_calculator", haversine)
        circum_points = kwargs.get("circum_points", 4)

        distances = distance_calculator(
            point[0],
            point[1],
            grid[0],
            grid[1],
        ).T[np.newaxis, :, :]

        min_idx = np.unravel_index(distances.argmin(), distances.shape)

        output = np.zeros_like(distances)
        output[
            min_idx[0],
            min_idx[1] - circum_points : min_idx[1] + circum_points + 1,
            min_idx[2] - circum_points : min_idx[2] + circum_points + 1,
        ] = 1

        return output

    def get_mask_series(self, timesteps, **kwargs):
        # read in parameters if submitted, otherwise use defaults
        masktype = kwargs.get("masktype", "rad")

        if masktype == "rad":
            get_mask = self.get_radmask
        elif masktype == "rect":
            get_mask = self.get_rectmask
        else:
            raise ValueError(f"Unsupported mask type {masktype}")

        # Generate grid
        ll_gridder(**kwargs)
        num_levels = kwargs.get("num_levels", 1)

        mask = None

        for stamp in timesteps:
            point = self.track[0][self.timestamps == stamp][0]

            temp_mask = get_mask(point, **kwargs)
            if num_levels > 1:
                temp_mask = np.tile(temp_mask, (num_levels, 1, 1))[np.newaxis, :, :, :]

            if mask is None:
                mask = temp_mask
            else:
                mask = np.vstack([mask, temp_mask])
        return mask

    def add_var_from_dataset(self, data, **kwargs):
        """
        Function to add data from a dataset to the track object.

        The function first checks if a dataset does not exist in the
        object, and if it does not, the function checks if a dataset
        exists on disk. If it does, the function loads the dataset and
        then checks to see if the input data variables are already present
        in the dataset. If they are, the function skips the variable.
        If they are not, the variable is added to a list of variables to
        process.

        If the dataset exists in the object, the function checks to see
        whether the input data variables are already present in the
        dataset. If they are, the function skips the variable. If they
        are not, the variable is added to a list of variables to process.

        If the list of variables to process is not empty, the function
        loops through the list and adds the variables to the dataset.
        The dataset is then stored in the object and saved to disk.

        Parameters
        ----------
        data : xr.Dataset, required
            Dataset containing the data to add to the track object

        Returns
        -------
        None

        """

        assert (
            type(data) == xr.Dataset
        ), f"Invalid data type {type(data)}. Expected xarray Dataset"

        # Determine if the data to be added is single or multilevel
        level_type = "S" if len(data.dims) == 3 else "M"
        ds_type = kwargs.get("masktype", "rad")

        # Check if the dataset doesn't exist in the object
        if not hasattr(self, f"{ds_type}_{level_type}_ds"):
            # Check if the dataset exists on disk
            if os.path.exists(self.filepath + f"{self.uid}.{ds_type}.{level_type}.nc"):
                # If it does, load the dataset
                print("Loading dataset...")
                setattr(
                    self,
                    f"{ds_type}_{level_type}_ds",
                    xr.open_dataset(
                        self.filepath + f"{self.uid}.{ds_type}.{level_type}.nc"
                    ),
                )

        # Check if the dataset exists in the object after checking disk
        if hasattr(self, f"{ds_type}_{level_type}_ds"):
            # Check if the data variables are already in the dataset
            for var in data.data_vars:
                if var in self.__getattribute__(f"{ds_type}_{level_type}_ds").data_vars:
                    print(f"Variable {var} already in dataset. Skipping...")
                else:
                    print(f"Adding variable {var} to processing list...")
                    if not hasattr(self, "var_list"):
                        self.var_list = [var]
                    else:
                        self.var_list.append(var)
        else:
            # If the dataset doesn't exist in the object, add all variables
            print("Creating dataset...")

            # Sanitize timestamps because ibtracs includes unsual time steps,
            # e.g. 603781 (Katrina, 2005) includes 2005-08-25 22:30:00,
            # 2005-08-29 11:10:00, 2005-08-29 14:45:00
            valid_steps = self.timestamps[np.isin(self.timestamps, data.time.values)]
            data_steps = data.sel(time=valid_steps)
            attrs = data_steps.attrs

            # Generate lat and lot vectors
            lat_vector, lon_vector = axis_generator(**kwargs)
            assert (
                lon_vector.shape <= data_steps.lon.shape
            ), f"Longitude vector is too long. Expected <={data_steps.lon.shape} but got {lon_vector.shape}. Downscaling not yet supported."
            assert (
                lat_vector.shape <= data_steps.lat.shape
            ), f"Latitude vector is too long. Expected <={data_steps.lat.shape} but got {lat_vector.shape}. Downscaling not yet supported."

            # Generate empty array to cast data with
            casting_array = xr.DataArray(
                np.NaN,
                dims=["lat", "lon"],
                coords={"lat": lat_vector, "lon": lon_vector},
            )
            regridder = xe.Regridder(
                data_steps,
                casting_array,
                "bilinear",
                parallel=True,
            )
            data_steps = regridder(data_steps)
            mask = self.get_mask_series(valid_steps, **kwargs)

            data_steps = data_steps.where(mask)
            data_steps.attrs = attrs

            setattr(self, f"{ds_type}_{level_type}_ds", data_steps)

            data_steps.to_netcdf(
                self.filepath + f"{self.uid}.{ds_type}.{level_type}.nc",
                mode="w",
                compute=True,
            )

        # If the list of variables to process is not empty, process them
        if hasattr(self, "var_list"):
            # Sanitize timestamps because ibtracs includes unsual time steps,
            # e.g. 603781 (Katrina, 2005) includes 2005-08-25 22:30:00,
            # 2005-08-29 11:10:00, 2005-08-29 14:45:00
            valid_steps = self.timestamps[np.isin(self.timestamps, data.time.values)]
            data_steps = data.sel(time=valid_steps)

            # Generate lat and lot vectors
            lat_vector, lon_vector = axis_generator(**kwargs)

            # Generate mask
            mask = self.get_mask_series(valid_steps, **kwargs)
            assert (
                lon_vector.shape <= data_steps.lon.shape
            ), f"Longitude vector is too long. Expected <={data_steps.lon.shape} but got {lon_vector.shape}. Downscaling not yet supported."
            assert (
                lat_vector.shape <= data_steps.lat.shape
            ), f"Latitude vector is too long. Expected <={data_steps.lat.shape} but got {lat_vector.shape}. Downscaling not yet supported."

            for var in self.var_list:
                print(f"Adding {var} to dataset...")
                var_steps = data_steps[var]
                # Generate empty array to cast data with
                casting_array = xr.DataArray(
                    np.NaN,
                    dims=["lat", "lon"],
                    coords={"lat": lat_vector, "lon": lon_vector},
                )
                regridder = xe.Regridder(
                    var_steps,
                    casting_array,
                    "bilinear",
                    parallel=True,
                )
                var_steps = regridder(var_steps)
                var_steps = var_steps.where(mask)

                # Add the variable to the dataset
                self.__getattribute__(f"{ds_type}_{level_type}_ds")[var] = var_steps

            self.__getattribute__(f"{ds_type}_{level_type}_ds").to_netcdf(
                self.filepath + f"{self.uid}.{ds_type}.{level_type}.appended.nc",
                mode="w",
                compute=True,
            )

            # Workaround to rewrite file with appended file
            self.__getattribute__(f"{ds_type}_{level_type}_ds").close()
            # Overwrite the original file with the appended file
            subprocess.run(
                [
                    "mv",
                    "-f",
                    f"{self.filepath}{self.uid}.{ds_type}.{level_type}.appended.nc",
                    f"{self.filepath}{self.uid}.{ds_type}.{level_type}.nc",
                ]
            )

            # remove the var_list attribute
            delattr(self, "var_list")
