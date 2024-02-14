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
import gc

# Base Libraries
import pandas as pd
import numpy as np
import xarray as xr
import xesmf as xe
import joblib as jl
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl.patch import geos_to_path
import itertools
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib as mpl


# TCBench Libraries
try:
    from utils import constants, data_lib
except:
    import constants, data_lib

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


def get_coord_vars(dataset):
    lon_coord = lat_coord = time_coord = level_coord = False
    coord_array = np.array(dataset.coords)

    for idx, coord in enumerate(coord_array):

        # retrieve the longitude coordinate if available
        if not lon_coord:
            lon_coord = (
                coord_array[idx]
                if np.any(
                    [
                        valid_name in coord
                        for valid_name in constants.valid_coords["longitude"]
                    ]
                )
                else False
            )

        if not lat_coord:
            lat_coord = (
                coord_array[idx]
                if np.any(
                    [
                        valid_name in coord
                        for valid_name in constants.valid_coords["latitude"]
                    ]
                )
                else False
            )

        if not time_coord:
            time_coord = (
                coord_array[idx]
                if np.any(
                    [
                        valid_name == coord
                        for valid_name in constants.valid_coords["time"]
                    ]
                )
                else False
            )

        if not level_coord:
            level_coord = (
                coord_array[idx]
                if np.any(
                    [
                        valid_name in coord
                        for valid_name in constants.valid_coords["level"]
                    ]
                )
                else False
            )

    assert np.all(
        [bool(lon_coord), bool(lat_coord), bool(time_coord)]
    ), "Missing lat, lon, or time coordinates in dataset"

    return lat_coord, lon_coord, time_coord, level_coord


# %% Data Processing / Preprocessing Library
"""
This cell contains the following classes and functions:
    
"""


def read_hist_track_file(**kwargs):
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
    tracks_path = kwargs.get(
        "tracks_path",
        f"/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/tracks/ibtracs/",
    )
    backend = kwargs.get("backend", pd.read_csv)
    track_cols = kwargs.get("track_cols", constants.ibtracs_cols)
    skip_rows = kwargs.get("skip_rows", [1])
    lon_mode = kwargs.get("lon_mode", 360)
    file_list = os.listdir(tracks_path)

    assert len(file_list) == 1, f"{tracks_path} has more than one file. Aborting."

    with open(f"{tracks_path}/{file_list[0]}") as handle:
        if backend == pd.read_csv:
            data = backend(
                handle,
                usecols=track_cols.get_colnames(),
                dtype=track_cols.get_dtypes(),
                skiprows=skip_rows,
                # parse_dates=track_cols.get_datetime_cols(),
                na_filter=False,  # Otherwise pandas interprets 'NA' as NaN
            )
            data["ISO_TIME"] = pd.to_datetime(data["ISO_TIME"])



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

    
    if lon_mode == 360:
        data["LON"] = data["LON"].apply(lambda x: x + 360 if x < 0 else x)
    elif lon_mode == 180:
        data["LON"] = data["LON"].apply(lambda x: x - 180 if x > 180 else x)
    else:
        raise ValueError(f"Unsupported lon_mode {lon_mode}")

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

    return valid_steps


def get_regrider(dataset: xr.Dataset, lat_coord: str, lon_coord: str, **kwargs):
    # Generate lat and lot vectors
    lat_vector, lon_vector = axis_generator(**kwargs)

    assert (
        lon_vector.shape <= dataset[lon_coord].shape
    ), f"Longitude vector is too long. Expected <={dataset[lon_coord].shape} but got {lon_vector.shape}. Downscaling not yet supported."
    assert (
        lat_vector.shape <= dataset[lat_coord].shape
    ), f"Latitude vector is too long. Expected <={dataset[lat_coord].shape} but got {lat_vector.shape}. Downscaling not yet supported."

    # If either the lat or lon vectors are smaller than the dataset, regrid
    if (
        lon_vector.shape != dataset[lon_coord].shape
        or lat_vector.shape != dataset[lat_coord].shape
    ):
        print("Making a regridder...")
        # Generate empty array to cast data with
        casting_array = xr.DataArray(
            np.NaN,
            dims=[lat_coord, lon_coord],
            coords={lat_coord: lat_vector, lon_coord: lon_vector},
        )
        regridder = xe.Regridder(
            dataset,
            casting_array,
            "bilinear",
        )

        return regridder(dataset)
    else:
        return None


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
            self.track = (track,)[0]  # Weird workaround - look into fixing
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

                # # For each type of data, load the data into the object
                # # Files follow <SID>.<TYPE>.<(S)ingle or (M)ultilevel>.nc naming convention
                # for file in file_list:
                #     # Get the variable name from the file name
                #     ds_type = file.split(".")[1]
                #     level_type = file.split(".")[2]
                #     # Load the data into the object
                #     setattr(
                #         self,
                #         ds_type + level_type + "ds",
                #         xr.open_dataset(self.filepath + file),
                #     )

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

    def get_mask(self, timestamp, **kwargs):
        # read in parameters if submitted, otherwise use defaults
        masktype = kwargs.get("masktype", "rad")

        if masktype == "rad":
            mask_getter = self.get_radmask
        elif masktype == "rect":
            mask_getter = self.get_rectmask
        else:
            raise ValueError(f"Unsupported mask type {masktype}")

        mask = None

        point = self.track[self.timestamps == timestamp][0]

        temp_mask = mask_getter(point, **kwargs)

        # Tile the mask if the number of levels is greater than 1
        num_levels = kwargs.get("num_levels", 1)
        if num_levels > 1:
            temp_mask = np.tile(temp_mask, (num_levels, 1, 1))[np.newaxis, :, :, :]

        if mask is None:
            mask = temp_mask
        else:
            mask = np.vstack([mask, temp_mask])

        # flip lat because reasons
        if num_levels > 1:
            mask = mask[:,:,::-1,:]
        # Squeeze the mask to remove unnecessary dimensions. Flip y because reasons?
        mask = mask.squeeze()[::-1, :]

        return mask

    def get_mask_series(self, timesteps, **kwargs):
        # read in parameters if submitted, otherwise use defaults
        masktype = kwargs.get("masktype", "rad")

        if masktype == "rad":
            get_mask = self.get_radmask
        elif masktype == "rect":
            get_mask = self.get_rectmask
        else:
            raise ValueError(f"Unsupported mask type {masktype}")

        num_levels = kwargs.get("num_levels", 1)

        mask = None

        for stamp in timesteps:
            point = self.track[0][self.timestamps == stamp][0]

            temp_mask = get_mask(point, **kwargs)
            print(num_levels)

            if num_levels > 1:
                temp_mask = np.tile(temp_mask, (num_levels, 1, 1))[np.newaxis, :, :, :]

            if mask is None:
                mask = temp_mask
            else:
                mask = np.vstack([mask, temp_mask])

        return mask

    def process_data_collection(self, data_collection, **kwargs):
        # Function to add all variables from a data collection object
        # to the track object
        assert isinstance(
            data_collection, data_lib.Data_Collection
        ), f"Invalid data type {type(data_collection)}. Expected data_lib.data_collection"

        # make a set of the years present in the tc_track timestamps
        years = sorted(list(set(self.timestamps.astype("datetime64[Y]").astype(str))))
        years = [int(year) for year in years]

        for var_type in data_collection.meta_dfs:
            # Skip the var type if it's empty
            if len(data_collection.meta_dfs[var_type].columns) == 0:
                (f"Skipping {var_type} because it's empty")
                continue

            ignored_vars = kwargs.get("ignore_vars", None)
            if ignored_vars:
                assert isinstance(
                    ignored_vars, list
                ), f"Invalid type for ignored_vars: {type(ignored_vars)}. Expected list"

            var_list = data_collection.meta_dfs[var_type].index
            if ignored_vars:
                var_list = var_list[~var_list.isin(ignored_vars)]
            var_meta = data_collection.meta_dfs[var_type].loc[var_list]

            for year in years:
                year = str(year)
                # Assert each variable in the collection is available for each
                # year the storm is active
                assert (
                    len(var_meta[year]) == var_meta[year].sum()
                ), f"Missing variable data for {year}"

            for var in var_list:

                ##TODO: check if file exsits and if it does, check if the variable is already in the file
                # if it is, skip it

                print("Years", years, "Variable", var)
                # Load the data for each variable
                var_data, _ = data_collection.retrieve_ds(
                    vars=var, years=years, **kwargs
                )

                print(var_data)

                # get coordinate names
                lat_coord, lon_coord, time_coord, level_coord = get_coord_vars(var_data)

                print(f"Adding {var} to {self.uid}...")

                # Sanitize timestamps because ibtracs includes unsual time steps
                valid_steps = sanitize_timestamps(self.timestamps, var_data)

                print(f"Processing time steps in parallel...")
                # Loop through each time step in parallel
                jl.Parallel(n_jobs=-1, verbose=2)(
                    jl.delayed(self.process_timestep)(
                        var_data.sel(time=t).copy(),
                        **{
                            "lat_coord": lat_coord,
                            "lon_coord": lon_coord,
                            "time_coord": time_coord,
                            "level_coord": level_coord,
                        },
                        timestamp=t,
                        **kwargs,
                    )
                    for t in valid_steps
                )

                # Garbage collect to free up memory
                gc.collect()

                # Read in the temporary files and merge them into the final file
                file_list = os.listdir(self.filepath)

                datavar = list(var_data.data_vars)[0]

                # Filter out files that do not include the temp preix, variable name, and UID
                for file in file_list.copy():
                    if "." + datavar + "." not in file:
                        file_list.remove(file)
                        print(
                            f"Removing {file} from file list because it doesn't include {datavar}"
                        )
                    elif "temp" not in file:
                        file_list.remove(file)
                        print(
                            f"Removing {file} from file list because it doesn't include temp"
                        )
                    elif self.uid not in file:
                        file_list.remove(file)
                        print(
                            f"Removing {file} from file list because it doesn't include {self.uid}"
                        )

                print("after removal: ", file_list)

                # If temporary files are present, merge them into the final file
                if len(file_list) > 0:
                    # Load the temporary files into an xarray dataset
                    temp_ds = xr.open_mfdataset(
                        [self.filepath + file for file in file_list],
                        concat_dim=time_coord,
                        combine="nested",
                        parallel=True,
                    )
                    print(temp_ds)

                    # calculate the encoding for the final dataset
                    encoding = {}
                    for data_var in temp_ds.data_vars:
                        encoding[data_var] = {
                            "original_shape": temp_ds[data_var].shape,
                            "_FillValue": temp_ds[data_var].encoding.get(
                                "_FillValue", -32767
                            ),
                            "dtype": kwargs.get("dtype", np.int16),
                            "add_offset": temp_ds[data_var].mean().compute().values,
                            "scale_factor": temp_ds[data_var].std().compute().values
                            / kwargs.get("scale_divisor", 2000),
                        }

                    # if the UID dataset exists, merge the new data into it
                    if os.path.exists(
                        self.filepath + f"{self.uid}.{kwargs.get('masktype', 'rad')}.nc"
                    ):
                        print(f"Merging {var} into {self.uid}...")
                        # Load the UID dataset
                        final_ds = xr.open_dataset(
                            self.filepath
                            + f"{self.uid}.{kwargs.get('masktype', 'rad')}.nc"
                        )

                        # Merge the new data into the final dataset
                        final_ds = xr.merge([final_ds, temp_ds])

                        # Save the final dataset to disk
                        final_ds.to_netcdf(
                            self.filepath
                            + f"{self.uid}.{kwargs.get('masktype', 'rad')}.appended.nc",
                            mode="w",
                            compute=True,
                            encoding=encoding,
                        )

                        # Workaround to rewrite file with appended file
                        # Overwrite the original file with the appended file
                        subprocess.run(
                            [
                                "mv",
                                "-f",
                                f"{self.filepath}{self.uid}.{kwargs.get('masktype', 'rad')}.appended.nc",
                                f"{self.filepath}{self.uid}.{kwargs.get('masktype', 'rad')}.nc",
                            ]
                        )
                    else:
                        # Save the final dataset to disk
                        temp_ds.to_netcdf(
                            self.filepath
                            + f"{self.uid}.{kwargs.get('masktype', 'rad')}.nc",
                            mode="w",
                            compute=True,
                            encoding=encoding,
                        )

                    # Close the temporary dataset
                    temp_ds.close()
                    del temp_ds

                    # delete the temporary files
                    for file in file_list:
                        os.remove(self.filepath + file)

                    # Garbage collect to free up memory
                    gc.collect()

    def process_timestep(self, data, **kwargs):
        """
        Function to add a dataset to the track object. It expects
        a dataset served by the data collection object.


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
            Coords:
                'lat' or 'Lat' must be in the latitude dimension name
                'lon' or 'Lon' must be in the longitude dimension
                'time', 'Time', or 't' must be found as the time dimension
                'lev', 'Lev', 'pressure', or 'Pressure' must be in the level dimension if the dataset is multilevel

        Returns
        -------
        None

        """

        # Determine if the data will be rectanglular or radial, default rad
        ds_type = kwargs.get("masktype", "rad")

        # Get the coordinate names
        lon_coord = kwargs.get("lon_coord", None)
        lat_coord = kwargs.get("lat_coord", None)
        time_coord = kwargs.get("time_coord", None)
        level_coord = kwargs.get("level_coord", None)

        # Determine the number of levels
        if level_coord:
            num_levels = data[level_coord].shape[0]
        else:
            num_levels = 1

        # Get the timestamp from the kwargs
        timestamp = kwargs.get("timestamp", None)
        assert timestamp is not None, "Fatal error: timestamp not found in kwargs"

        var_list = None

        # Check if the dataset doesn't exist in the object
        if not hasattr(self, f"{ds_type}_ds"):
            # Check if the dataset exists on disk
            if os.path.exists(self.filepath + f"{self.uid}.{ds_type}.nc"):
                # If it does, load the dataset
                print("Loading dataset...")
                setattr(
                    self,
                    f"{ds_type}_ds",
                    xr.open_dataset(self.filepath + f"{self.uid}.{ds_type}.nc"),
                )

        # Check if the dataset exists in the object after checking disk
        if hasattr(self, f"{ds_type}_ds"):
            # Check if the data variables are already in the dataset
            for var in data.data_vars:
                if var in self.__getattribute__(f"{ds_type}_ds").data_vars:
                    print(f"Variable {var} already in dataset. Skipping...")
                else:
                    print(f"Adding variable {var} to processing list...")
                    if not var_list:
                        var_list = [var]
                    else:
                        var_list.append(var)
        elif var_list is None: # add all the datavars to the list
           var_list = list(data.data_vars)

        if var_list is not None:
            data = data[var_list]

            attrs = data.attrs

            # Retrieve regridder if necessary
            regridder = get_regrider(dataset=data, **kwargs)

            if regridder is not None:
                data = regridder(data)

            level_coord = kwargs.get("level_coord", None)
            if level_coord:
                num_levels = data[level_coord].shape[0]

            mask = self.get_mask(num_levels=num_levels, **kwargs)
            # mask = self.get_mask_series(valid_steps, **kwargs)

            data = data.where(mask)
            data.attrs = attrs

            setattr(self, f"{ds_type}_ds", data)

            encoding = {}
            for data_var in data.data_vars:
                encoding[data_var] = {
                    "original_shape": data[data_var].shape,
                    "_FillValue": data[data_var].encoding.get("_FillValue", -32767),
                    "dtype": kwargs.get("dtype", np.int16),
                    "add_offset": data[data_var].mean().compute().values,
                    "scale_factor": data[data_var].std().compute().values
                    / kwargs.get("scale_divisor", 2000),
                }

            data.to_netcdf(
                self.filepath
                + f"temp_{np.where(self.timestamps == timestamp)[0][0]}.{data_var}.{self.uid}.{ds_type}.nc",
                mode="w",
                compute=True,
                encoding=encoding,
            )

            data.close()
            del data
            delattr(self, f"{ds_type}_ds")
            gc.collect()



        # else:
        #     # If the dataset doesn't exist in the object, add all variables
        #     # print("Creating dataset...")

        #     attrs = data.attrs

        #     # Retrieve regridder if necessary
        #     regridder = get_regrider(dataset=data, **kwargs)

        #     if regridder is not None:
        #         data = regridder(data)

        #     level_coord = kwargs.get("level_coord", None)
        #     if level_coord:
        #         num_levels = data[level_coord].shape[0]

        #     mask = self.get_mask(num_levels=num_levels, **kwargs)
        #     # mask = self.get_mask_series(valid_steps, **kwargs)

        #     data = data.where(mask)
        #     data.attrs = attrs

        #     setattr(self, f"{ds_type}_ds", data)

        #     encoding = {}
        #     for data_var in data.data_vars:
        #         encoding[data_var] = {
        #             "original_shape": data[data_var].shape,
        #             "_FillValue": data[data_var].encoding.get("_FillValue", -32767),
        #             "dtype": kwargs.get("dtype", np.int16),
        #             "add_offset": data[data_var].mean().compute().values,
        #             "scale_factor": data[data_var].std().compute().values
        #             / kwargs.get("scale_divisor", 2000),
        #         }

        #     data.to_netcdf(
        #         self.filepath
        #         + f"temp_{np.where(self.timestamps == timestamp)[0][0]}.{data_var}.{self.uid}.{ds_type}.nc",
        #         mode="w",
        #         compute=True,
        #         encoding=encoding,
        #     )

        #     data.close()
        #     del data
        #     delattr(self, f"{ds_type}_ds")

        # ## TODO: update to work like part above
        # # If the list of variables to process is not empty, process them
        # if var_list:
        #     # Retrieve regridder if necessary
        #     regridder = get_regrider(dataset=data, **kwargs)

        #     if regridder is not None:
        #         data = regridder(data)

        #     # # Initiate the encoding dictionary so that it can be updated in the loop
        #     # encoding = {}
        #     # for var in self.__getattribute__(f"{ds_type}_ds").data_vars:
        #     #     print(var, self.__getattribute__(f"{ds_type}_ds")[var].encoding)
        #     #     encoding[var] = {
        #     #         "dtype": self.__getattribute__(f"{ds_type}_ds")[var].encoding[
        #     #             "dtype"
        #     #         ],
        #     #         "_FillValue": self.__getattribute__(f"{ds_type}_ds")[var].encoding[
        #     #             "_FillValue"
        #     #         ],
        #     #         "add_offset": self.__getattribute__(f"{ds_type}_ds")[var].encoding[
        #     #             "add_offset"
        #     #         ],
        #     #         "scale_factor": self.__getattribute__(f"{ds_type}_ds")[
        #     #             var
        #     #         ].encoding["scale_factor"],
        #     #     }

        #     for var in var_list:
        #         # print(f"Adding {var} to dataset...")
        #         var_steps = data[var]
        #         attrs = var_steps.attrs

        #         mask = self.get_mask(num_levels=num_levels, **kwargs)

        #         var_steps = var_steps.where(mask)

        #         # Ensure that the attributes didn't get stripped through processing
        #         var_steps.attrs = attrs

        #         # Add the variable to the dataset
        #         self.__getattribute__(f"{ds_type}_ds")[var] = var_steps

        #         encoding[var] = {
        #             "original_shape": var_steps.shape,
        #             "_FillValue": var_steps.encoding.get("_FillValue", -32767),
        #             "dtype": kwargs.get("dtype", np.int16),
        #             "add_offset": var_steps.mean().compute().values,
        #             "scale_factor": var_steps.std().compute().values
        #             / kwargs.get("scale_divisor", 2000),
        #         }

        #     # self.__getattribute__(f"{ds_type}_ds").to_netcdf(
        #     #     self.filepath + f"{self.uid}.{ds_type}.appended.nc",
        #     #     mode="w",
        #     #     compute=True,
        #     #     encoding=encoding,
        #     # )

        #     # # Workaround to rewrite file with appended file
        #     # self.__getattribute__(f"{ds_type}_ds").close()
        #     # # Overwrite the original file with the appended file
        #     # subprocess.run(
        #     #     [
        #     #         "mv",
        #     #         "-f",
        #     #         f"{self.filepath}{self.uid}.{ds_type}.appended.nc",
        #     #         f"{self.filepath}{self.uid}.{ds_type}.nc",
        #     #     ]
        #     # )

        #     # remove the var_list attribute
        #     del var_list
        #     delattr(self, f"{ds_type}_ds")
        #     gc.collect()

        # gc.collect()

    # Function to load the data from storage
    def load_data(self, **kwargs):
        """
        Function to load the data from storage

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        ds_type = kwargs.get("ds_type", "rad")

        # Check if the dataset doesn't exist in the object
        if not hasattr(self, f"{ds_type}_ds"):
            # Check if the dataset exists on disk
            if os.path.exists(self.filepath + f"{self.uid}.{ds_type}.nc"):
                # If it does, load the dataset
                print("Loading dataset...")
                setattr(
                    self,
                    f"{ds_type}_ds",
                    xr.open_dataset(self.filepath + f"{self.uid}.{ds_type}.nc"),
                )
            else:
                print(f"Dataset {self.uid}.{ds_type}.nc not found on disk")
                print(
                    'You can create the dataset by running the "process_data_collection" method'
                    " with a data collection object as an argument."
                )

    def plotTrack(self, **kwargs):
        """

        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
            The default is (6, 2). Note that the tuple is fed inverted to the
            matplotlib figure function, i.e., default is fed as (2,6)

        height : float, optional
            Height at which to plot the track and points.
            The default is 0.0.

        point_size : int, optional
            Size of the points. The default is 150.

        step : int, optional
            Timesteps to skip when plotting points. The default is 6.

        radius : int, optional
            The number of degrees east/west/north/south of the track minima/maxima
            to plot. The default is 20.

        background_color : tuple, optional
            Background color. The default is matplotlibs tab:cyan with 0.25 alpha.

        land_color : tuple, optional
            Land color. The default is matplotlibs tab:green with 0.05 alpha.

        coastline_color: str, optional
            Coastline color. The default is "grey".

        view_angles : tuple, optional
            View angles. The default is (25, -90).

        track_color : str, optional
            Track color. The default is "black".

        point_color : tuple, optional
            Point color. The default is matplotlibs tab:orange.



        Sources:
        https://stackoverflow.com/questions/30223161/how-to-increase-the-size-of-an-axis-stretch-in-a-3d-plot
        """

        width, height = kwargs.get("figsize", (6, 2))
        point_height = kwargs.get("height", 0.0)
        point_size = kwargs.get("point_size", 150)
        step = kwargs.get("step", 6)
        radius = kwargs.get("radius", 20)
        background_color = kwargs.get(
            "background_color",
            (*mpl.colors.to_rgb(mpl.colors.TABLEAU_COLORS["tab:cyan"]), 0.25),
        )
        land_color = kwargs.get(
            "land_color",
            (*mpl.colors.to_rgb(mpl.colors.TABLEAU_COLORS["tab:green"]), 0.05),
        )
        coastline_color = kwargs.get("coastline_color", "grey")
        view_angles = kwargs.get("view_angles", (25, -90))
        track_color = kwargs.get("track_color", "black")
        point_color = kwargs.get(
            "point_color", mpl.colors.to_rgb(mpl.colors.TABLEAU_COLORS["tab:orange"])
        )

        xlims = (self.track[:, 1].min() - radius, self.track[:, 1].max() + radius)
        ylims = (self.track[:, 0].min() - radius, self.track[:, 0].max() + radius)

        fig = plt.figure(figsize=(height, width), dpi=150)
        ax3d = fig.add_axes([0, 0, width, height], projection="3d")

        # Make an axes that we can use for mapping the data in 2d.
        proj_ax = plt.figure().add_axes(
            [0, 0, width, height], projection=ccrs.PlateCarree()
        )

        proj_ax.autoscale_view()

        ax3d.view_init(*view_angles)
        ax3d.set_xlim(xlims)
        ax3d.set_ylim(ylims)
        # ax3d.set_xlim(*proj_ax.get_xlim())
        # ax3d.set_ylim(*proj_ax.get_ylim())
        ax3d.set_zlim(0, 0.5)
        ax3d.set_box_aspect((width, height * 2, 0.5))
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.xaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax3d.yaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax3d.zaxis.set_pane_color(background_color)

        ax3d.grid(False)
        ax3d.zaxis.line.set_lw(0.0)
        ax3d.set_zticks([])

        ax3d.scatter3D(
            self.track[::step, 1],
            self.track[::step, 0],
            np.ones_like(self.track[::step, 0]) * point_height,
            c=point_color,
            s=point_size,
            marker="2",
        )
        ax3d.plot3D(
            self.track[:, 1],
            self.track[:, 0],
            np.ones_like(self.track[:, 0]) * point_height,
            c=track_color,
            # alpha=0.5,
        )
        # Now add coastlines.
        concat = lambda iterable: list(itertools.chain.from_iterable(iterable))

        target_projection = proj_ax.projection

        feature = cartopy.feature.NaturalEarthFeature("physical", "land", "110m")
        geoms = feature.geometries()

        proj_ax.set_xlim(xlims)
        proj_ax.set_ylim(ylims)

        # Use the convenience (private) method to get the extent as a shapely geometry.
        boundary = proj_ax._get_extent_geom()

        # Transform the geometries from PlateCarree into the desired projection.
        geoms = [
            target_projection.project_geometry(geom, feature.crs) for geom in geoms
        ]
        # Clip the geometries based on the extent of the map (because mpl3d can't do it for us)
        geoms = [boundary.intersection(geom) for geom in geoms]

        # Convert the geometries to paths so we can use them in matplotlib.
        paths = concat(geos_to_path(geom) for geom in geoms)
        polys = concat(path.to_polygons() for path in paths)
        lc = PolyCollection(
            polys,
            edgecolor=coastline_color,
            facecolor=land_color,
            closed=True,
        )
        ax3d.add_collection3d(lc, zs=0)

        plt.close(proj_ax.figure)
        plt.show()

    def plot3D(self, var, timestamps, **kwargs):
        """
        Function to plot 3D data on a map, showing the variable in 3D for the given timestamps

        Based off of:
        https://stackoverflow.com/questions/13570287/image-overlay-in-3d-plot
        https://stackoverflow.com/questions/23785408/3d-cartopy-similar-to-matplotlib-basemap
        https://stackoverflow.com/questions/48269014/contourf-in-3d-cartopy%5D
        Parameters
        ----------
        var : STR, required
            Variable to plot

        timestamp : STR, required
            Timestamp to plot

        Returns
        -------
        None

        """

        # Check if the dataset doesn't exist in the object
        if not hasattr(self, f"{kwargs.get('ds_type', 'rad')}_ds"):
            print("Data not yet loaded - trying to load it now...")
            self.load_data(**kwargs)

        data = getattr(self, f"{kwargs.get('ds_type', 'rad')}_ds")[var]

        # def f(x, y):
        #     x, y = np.meshgrid(x, y)
        #     return (1 - x / 2 + x**5 + y**3 + x * y**2) * np.exp(-(x**2) - y**2)

        # nx, ny = 256, 512
        # X = np.linspace(-180, 10, nx)
        # Y = np.linspace(-90, 90, ny)
        # Z = f(np.linspace(-3, 3, nx), np.linspace(-3, 3, ny))

        fig, ax3d = plt.subplots(subplot_kw={"projection": "3d"})
        # ax3d = fig.add_axes([0, 0, 1, 1], projection="3d")

        # Make an axes that we can use for mapping the data in 2d.
        # proj_ax = plt.figure().add_axes([0, 0, 1, 1], projection=ccrs.Mercator())

        lat_coord, lon_coord, time_coord, level_coord = get_coord_vars(data)

        for timestamp in timestamps:
            minn, maxx = (
                data.sel({time_coord: timestamp}).min().values,
                data.sel({time_coord: timestamp}).max().values,
            )
            if level_coord:
                for level in data[level_coord].values:
                    z = int(level)

                    level_data = data.sel({time_coord: timestamp, level_coord: level})
                    level_data = level_data.where(~level_data.isnull(), drop=True)

                    # minn, maxx = color_data.min(), color_data.max()
                    norm = mpl.colors.Normalize(vmin=minn, vmax=maxx)
                    m = plt.cm.ScalarMappable(norm=norm, cmap="seismic")
                    m.set_array([])
                    fcolors = m.to_rgba(level_data.values, alpha=0.10)

                    X, Y = np.meshgrid(
                        level_data[lon_coord].values, level_data[lat_coord].values[::-1]
                    )

                    ax3d.plot_surface(
                        X,
                        Y,
                        np.ones(X.shape) * z,
                        facecolors=fcolors,
                        vmin=minn,
                        vmax=maxx,
                        shade=False,
                        # transform=ccrs.PlateCarree(),
                        # alpha=0.05,
                    )
                fig.colorbar(m, ax=ax3d, orientation="vertical")
                ax3d.invert_zaxis()
                # fig2, ax2 = plt.subplots()
                # ax2.imshow(color_data, cmap="viridis")

        # cs = proj_ax.contourf(X, Y, Z, transform=ccrs.PlateCarree(), alpha=0.4)

        # for zlev, collection in zip(cs.levels, cs.collections):
        #     paths = collection.get_paths()
        #     # Figure out the matplotlib transform to take us from the X, Y coordinates
        #     # to the projection coordinates.
        #     trans_to_proj = collection.get_transform() - proj_ax.transData

        #     paths = [trans_to_proj.transform_path(path) for path in paths]
        #     verts3d = [
        #         np.concatenate(
        #             [path.vertices, np.tile(zlev, [path.vertices.shape[0], 1])], axis=1
        #         )
        #         for path in paths
        #     ]
        #     codes = [path.codes for path in paths]
        #     pc = Poly3DCollection([])
        #     pc.set_verts_and_codes(verts3d, codes)

        #     # Copy all of the parameters from the contour (like colors) manually.
        #     # Ideally we would use update_from, but that also copies things like
        #     # the transform, and messes up the 3d plot.
        #     pc.set_facecolor(collection.get_facecolor())
        #     pc.set_edgecolor(collection.get_edgecolor())
        #     pc.set_alpha(collection.get_alpha())

        #     ax3d.add_collection3d(pc)

        # proj_ax.autoscale_view()

        # ax3d.set_xlim(*proj_ax.get_xlim())
        # ax3d.set_ylim(*proj_ax.get_ylim())
        # ax3d.set_zlim(data[level_coord].min().values, data[level_coord].max().values)

        # # Now add coastlines.
        # concat = lambda iterable: list(itertools.chain.from_iterable(iterable))

        # target_projection = proj_ax.projection

        # feature = cartopy.feature.NaturalEarthFeature("physical", "land", "110m")
        # geoms = feature.geometries()

        # # Use the convenience (private) method to get the extent as a shapely geometry.
        # boundary = proj_ax._get_extent_geom()

        # # Transform the geometries from PlateCarree into the desired projection.
        # geoms = [
        #     target_projection.project_geometry(geom, feature.crs) for geom in geoms
        # ]
        # # Clip the geometries based on the extent of the map (because mpl3d can't do it for us)
        # geoms = [boundary.intersection(geom) for geom in geoms]

        # # Convert the geometries to paths so we can use them in matplotlib.
        # paths = concat(geos_to_path(geom) for geom in geoms)
        # polys = concat(path.to_polygons() for path in paths)
        # lc = PolyCollection(polys, edgecolor="black", facecolor="green", closed=True)
        # ax3d.add_collection3d(lc, zs=ax3d.get_zlim()[0])

        # plt.close(proj_ax.figure)
        # plt.show()

        pass

    # # This will be eliminated later
    # def add_var_from_dataset(self, data, **kwargs):
    #     """
    #     Function to add data from a dataset to the track object.

    #     The function first checks if a dataset does not exist in the
    #     object, and if it does not, the function checks if a dataset
    #     exists on disk. If it does, the function loads the dataset and
    #     then checks to see if the input data variables are already present
    #     in the dataset. If they are, the function skips the variable.
    #     If they are not, the variable is added to a list of variables to
    #     process.

    #     If the dataset exists in the object, the function checks to see
    #     whether the input data variables are already present in the
    #     dataset. If they are, the function skips the variable. If they
    #     are not, the variable is added to a list of variables to process.

    #     If the list of variables to process is not empty, the function
    #     loops through the list and adds the variables to the dataset.
    #     The dataset is then stored in the object and saved to disk.

    #     Parameters
    #     ----------
    #     data : xr.Dataset, required
    #         Dataset containing the data to add to the track object

    #     Returns
    #     -------
    #     None

    #     """

    #     assert (
    #         type(data) == xr.Dataset
    #     ), f"Invalid data type {type(data)}. Expected xarray Dataset"

    #     # Determine if the data to be added is single or multilevel
    #     level_type = "S" if len(data.dims) == 3 else "M"
    #     ds_type = kwargs.get("masktype", "rad")

    #     # Check if the dataset doesn't exist in the object
    #     if not hasattr(self, f"{ds_type}_{level_type}_ds"):
    #         # Check if the dataset exists on disk
    #         if os.path.exists(self.filepath + f"{self.uid}.{ds_type}.{level_type}.nc"):
    #             # If it does, load the dataset
    #             print("Loading dataset...")
    #             setattr(
    #                 self,
    #                 f"{ds_type}_{level_type}_ds",
    #                 xr.open_dataset(
    #                     self.filepath + f"{self.uid}.{ds_type}.{level_type}.nc"
    #                 ),
    #             )

    #     # Check if the dataset exists in the object after checking disk
    #     if hasattr(self, f"{ds_type}_{level_type}_ds"):
    #         # Check if the data variables are already in the dataset
    #         for var in data.data_vars:
    #             if var in self.__getattribute__(f"{ds_type}_{level_type}_ds").data_vars:
    #                 print(f"Variable {var} already in dataset. Skipping...")
    #             else:
    #                 print(f"Adding variable {var} to processing list...")
    #                 if not hasattr(self, "var_list"):
    #                     self.var_list = [var]
    #                 else:
    #                     self.var_list.append(var)
    #     else:
    #         # If the dataset doesn't exist in the object, add all variables
    #         print("Creating dataset...")

    #         # Sanitize timestamps because ibtracs includes unsual time steps,
    #         # e.g. 603781 (Katrina, 2005) includes 2005-08-25 22:30:00,
    #         # 2005-08-29 11:10:00, 2005-08-29 14:45:00
    #         valid_steps = self.timestamps[np.isin(self.timestamps, data.time.values)]
    #         data_steps = data.sel(time=valid_steps)
    #         attrs = data_steps.attrs

    #         # Generate lat and lot vectors
    #         lat_vector, lon_vector = axis_generator(**kwargs)
    #         assert (
    #             lon_vector.shape <= data_steps.lon.shape
    #         ), f"Longitude vector is too long. Expected <={data_steps.lon.shape} but got {lon_vector.shape}. Downscaling not yet supported."
    #         assert (
    #             lat_vector.shape <= data_steps.lat.shape
    #         ), f"Latitude vector is too long. Expected <={data_steps.lat.shape} but got {lat_vector.shape}. Downscaling not yet supported."

    #         # Generate empty array to cast data with
    #         casting_array = xr.DataArray(
    #             np.NaN,
    #             dims=["lat", "lon"],
    #             coords={"lat": lat_vector, "lon": lon_vector},
    #         )
    #         regridder = xe.Regridder(
    #             data_steps,
    #             casting_array,
    #             "bilinear",
    #             # parallel=True,
    #         )
    #         data_steps = regridder(data_steps)
    #         mask = self.get_mask_series(valid_steps, **kwargs)

    #         data_steps = data_steps.where(mask)
    #         data_steps.attrs = attrs

    #         setattr(self, f"{ds_type}_{level_type}_ds", data_steps)

    #         data_steps.to_netcdf(
    #             self.filepath + f"{self.uid}.{ds_type}.{level_type}.nc",
    #             mode="w",
    #             compute=True,
    #         )

    #     # If the list of variables to process is not empty, process them
    #     if hasattr(self, "var_list"):
    #         # Sanitize timestamps because ibtracs includes unsual time steps,
    #         # e.g. 603781 (Katrina, 2005) includes 2005-08-25 22:30:00,
    #         # 2005-08-29 11:10:00, 2005-08-29 14:45:00
    #         valid_steps = self.timestamps[np.isin(self.timestamps, data.time.values)]
    #         data_steps = data.sel(time=valid_steps)

    #         # Generate lat and lot vectors
    #         lat_vector, lon_vector = axis_generator(**kwargs)

    #         # Generate mask
    #         mask = self.get_mask_series(valid_steps, **kwargs)
    #         assert (
    #             lon_vector.shape <= data_steps.lon.shape
    #         ), f"Longitude vector is too long. Expected <={data_steps.lon.shape} but got {lon_vector.shape}. Downscaling not yet supported."
    #         assert (
    #             lat_vector.shape <= data_steps.lat.shape
    #         ), f"Latitude vector is too long. Expected <={data_steps.lat.shape} but got {lat_vector.shape}. Downscaling not yet supported."

    #         for var in self.var_list:
    #             print(f"Adding {var} to dataset...")
    #             var_steps = data_steps[var]
    #             # Generate empty array to cast data with
    #             casting_array = xr.DataArray(
    #                 np.NaN,
    #                 dims=["lat", "lon"],
    #                 coords={"lat": lat_vector, "lon": lon_vector},
    #             )
    #             regridder = xe.Regridder(
    #                 var_steps,
    #                 casting_array,
    #                 "bilinear",
    #                 parallel=True,
    #             )
    #             var_steps = regridder(var_steps)
    #             var_steps = var_steps.where(mask)

    #             # Add the variable to the dataset
    #             self.__getattribute__(f"{ds_type}_{level_type}_ds")[var] = var_steps

    #         self.__getattribute__(f"{ds_type}_{level_type}_ds").to_netcdf(
    #             self.filepath + f"{self.uid}.{ds_type}.{level_type}.appended.nc",
    #             mode="w",
    #             compute=True,
    #         )

    #         # Workaround to rewrite file with appended file
    #         self.__getattribute__(f"{ds_type}_{level_type}_ds").close()
    #         # Overwrite the original file with the appended file
    #         subprocess.run(
    #             [
    #                 "mv",
    #                 "-f",
    #                 f"{self.filepath}{self.uid}.{ds_type}.{level_type}.appended.nc",
    #                 f"{self.filepath}{self.uid}.{ds_type}.{level_type}.nc",
    #             ]
    #         )

    #         # remove the var_list attribute
    #         delattr(self, "var_list")


# %%
