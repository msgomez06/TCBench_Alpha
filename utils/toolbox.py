#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:45:42 2023

This file contains a "toolbox" library - functions and constants that will be
used in other scripts to perform the tasks associated with TCBench

@author: mgomezd1
"""

#%% Imports and base paths

# OS and IO
import os
import traceback

# Base Libraries
import pandas as pd
import numpy as np
import xarray as xr

# TCBench Libraries
import constants

# Retrieve Repository Path
repo_path = '/'+os.path.join(*os.getcwd().split('/')[:-1])

print(repo_path)

#%% Auxilliary Functions

# Lat-lon grid generator
def ll_gridder(origin = (0,0),
               resolution = 0.25,
               lon_mode = 360,
               lat_limits = None,
               lon_limits = None,
               use_poles = True):
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
    assert type(use_poles) is bool, f'Unexpected type for use_poles:{type(use_poles)}. Expected boolean.'
    assert lon_mode == 180 or lon_mode == 360, f'Invalid long mode in ll_gridder: {lon_mode}. Expected 180 or 360 as ints'
    
    if lat_limits is not None:
        assert origin[0] >= lat_limits[0], 'Origin lower limit less than origin value'
        assert origin[0] <= lat_limits[1], 'Origin lower limit less than origin value'
    else:
        lat_limits = (-90,90)
    
    if lon_limits is not None:
        assert origin[1] >= lon_limits[0], 'Origin lower limit less than origin value'
        assert origin[1] <= lon_limits[1], 'Origin lower limit less than origin value'    
    else:
        lon_limits = (0,360) if lon_mode == 360 else (-180,180)
    
    lat_vector = np.hstack([ np.flip(np.arange(origin[0], lat_limits[0] - resolution*use_poles, -resolution)),
                             np.arange(origin[0], lat_limits[1] + resolution*use_poles, resolution),])
    
    if lon_mode == 360:
        lon_vector = np.hstack([ np.flip(np.arange(origin[1], lon_limits[0] - resolution, -resolution)),
                                 np.arange(origin[1], lon_limits[1], resolution),])
        
    else:
        lon_vector = np.hstack([ np.flip(np.arange(origin[1], lon_limits[0], -resolution)),
                                 np.arange(origin[1], lon_limits[1]+resolution, resolution),])
    
    # Sanitize outputs
    lat_vector = np.unique(lat_vector)
    lon_vector = np.unique(lon_vector)
    
    lats, lons = np.meshgrid(lat_vector, lon_vector)
    
    return lats, lons

    

# Great circle distance calculations
def haversine(latp, lonp, lat_list, lon_list):
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
    latp = np.radians(latp)
    lonp = np.radians(lonp)
    lat_list = np.radians(lat_list)
    lon_list = np.radians(lon_list)
    
    dlon = lonp - lon_list
    dlat = latp - lat_list
    a = np.power(np.sin(dlat / 2),2) + np.cos(lat_list) * np.cos(latp) * np.power(np.sin(dlon / 2), 2)
    return 2 * 6371 * np.arcsin(np.sqrt(a))

#%% Data Processing / Preprocessing Library
"""
This cell contains the following classes and functions:
    
"""
def read_hist_track_file(tracks_path = f'{repo_path}/tracks/ibtracs/',
                         backend = pd.read_csv,
                         track_cols = constants.ibtracs_cols,
                         skip_rows = [1],
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
    
    with open(f'{tracks_path}/{file_list[0]}') as handle:
        if backend == pd.read_csv:      

            data = backend(handle,
                           usecols = track_cols.get_colnames(),
                           dtype = track_cols.get_dtypes(),
                           skiprows = skip_rows,
                           parse_dates = track_cols.get_datetime_cols(),
                           #index_col = track_cols._track_cols__metadata['UID'],
                           )
            
    
    return data

def read_py_track_file(tracks_path = f'{repo_path}/tracks/sgstracks/',
                       year = 2000,
                       backend = pd.read_csv
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
        if str(year) not in file: file_list.remove(file)
        
    assert len(file_list) == 1, f"{tracks_path} has more than one file for year {year}. Aborting."
    
    with open(f'{tracks_path}/{file_list[0]}') as handle:
              data = backend(handle)
    
    return data

def process_py_track_data(data,
                          track_cols):
    
    for column in track_cols.__dict__.keys():
        assert column in data.columns, f'{column} not found in header columns'
        
        if type(track_cols.__dict__[column])==dict: print(column)
    
    # TODO: 
    
    pass

# Functions to sanitize timestamp data
def sanitize_timestamps(timestamps,
                        data):
    # Sanitize timestamps because ibtracs includes unsual time steps,
    # e.g. 603781 (Katrina, 2005) includes 2005-08-25 22:30:00, 
    # 2005-08-29 11:10:00, 2005-08-29 14:45:00
    valid_steps = timestamps[np.isin(timestamps, data.time.values)]
    
    data_steps = data.sel(time=valid_steps)
    return data_steps


# Class used to process track file and instantiate a track per event


# Class presenting each track type
class tc_track:
    def __init__(self,
                 UID,
                 NAME,
                 track,
                 timestamps,
                 ):
        try:
            assert type(UID) == str, f'Unique Identifier Error. {type(UID)} given when string was expected'
            assert type(NAME) == str, f'Name Error. {type(NAME)} given when string was expected'
            assert type(track) == np.ndarray, f'Track Error. {type(track)} given when numpy array was expected'
            assert (type(timestamps) == np.ndarray or type(timestamps) == pd.DataFrame), f'Unsupported timestamp iterator {type(timestamps)}'
            assert np.issubdtype(timestamps.dtype, np.datetime64), f'Unsupported timestamp type: {timestamps.dtype} - expected datetime64'
            
            self.uid = UID
            self.name = NAME
            self.track = track,
            self.timestamps = timestamps
        except Exception as e:
            print(f'Encountered and exception when instiating a tc_track object: \n{e}\n')
            print(traceback.format_exc())

    
    def get_varmask(self,
                    point,
                    **kwargs):
        
        # read in parameters if submitted, otherwise use defaults
        radius = kwargs['radius'] if 'radius' in kwargs.keys() else 500
        grid = kwargs['grid'] if 'grid' in kwargs.keys() else ll_gridder()
        distance_calculator = kwargs['distance_calculator'] if 'distance_calculator' in kwargs.keys() else haversine
        radius = kwargs['radius'] if 'radius' in kwargs.keys() else 500



        return np.flip((distance_calculator(point[0],
                                   point[1],
                                   grid[0],
                                   grid[1],
                                   )<=radius).T[np.newaxis,:,:],
                       axis=1)
    
    def get_mask_series(self,
                        timesteps,
                        **kwargs):
        
        # read in parameters if submitted, otherwise use defaults
        radius = kwargs['radius'] if 'radius' in kwargs.keys() else 500
        resolution = kwargs['resolution'] if 'resolution' in kwargs.keys() else 0.25
        origin = kwargs['origin'] if 'origin' in kwargs.keys() else (0,0)
        lon_mode = kwargs['lon_mode'] if 'lon_mode' in kwargs.keys() else 360
        lat_limits = kwargs['lat_limits'] if 'lat_limits' in kwargs.keys() else None
        lon_limits = kwargs['lon_limits'] if 'lon_limits' in kwargs.keys() else None
        use_poles = kwargs['use_poles'] if 'use_poles' in kwargs.keys() else True
        distance_calculator = kwargs['distance_calculator'] if 'distance_calculator' in kwargs.keys() else haversine
        
        # Generate grid
        grid = ll_gridder(resolution = resolution,
                          origin = origin,
                          lon_mode = lon_mode,
                          lat_limits=lat_limits,
                          lon_limits=lon_limits,
                          use_poles=use_poles)
        
        mask = None

        for stamp in timesteps:
            point = self.track[0][self.timestamps == stamp][0]
            
            temp_mask = self.get_varmask(point,
                                         radius=radius,
                                         grid=grid)
                
            if mask is None:
                mask = temp_mask
            else:
                mask = np.vstack([mask,
                                  temp_mask])
        return mask

    def add_var_from_dataarray(self,
                            radius,
                            data,
                            resolution = None,
                            origin = None
                            ):
        assert type(data) == xr.DataArray, f'Invalid data type {type(data)}. Expected xarray DataArray'
        
        # Sanitize timestamps because ibtracs includes unsual time steps,
        # e.g. 603781 (Katrina, 2005) includes 2005-08-25 22:30:00, 
        # 2005-08-29 11:10:00, 2005-08-29 14:45:00
        valid_steps = self.timestamps[np.isin(self.timestamps, data.time.values)]
        data_steps = data.sel(time=valid_steps)
        
        #TODO: sanitize data_steps shape to fit resolution


        
        mask = self.get_mask_series(valid_steps,
                                    radius=radius,
                                    resolution=resolution)

        
        data_steps = data_steps.where(mask)
        
        attr_name = f'{data.name}_res{resolution}_rad{radius}'
        
        setattr(self, 
                attr_name, 
                data_steps)
    
    def add_var_from_dataset(self,
                             radius,
                             data,
                             resolution = None,
                             origin = None):
        assert type(data) == xr.Dataset, f'Invalid data type {type(data)}. Expected xarray Dataset'

        # Sanitize timestamps because ibtracs includes unsual time steps,
        # e.g. 603781 (Katrina, 2005) includes 2005-08-25 22:30:00, 
        # 2005-08-29 11:10:00, 2005-08-29 14:45:00
        valid_steps = self.timestamps[np.isin(self.timestamps, data.time.values)]
        data_steps = data.sel(time=valid_steps)
        
        mask = self.get_mask_series(valid_steps,
                                    radius=radius,
                                    resolution=resolution)
        
        data_steps = data_steps.where(mask)

        setattr(self,
                'ds',
                data_steps)
    