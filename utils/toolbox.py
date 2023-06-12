#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:45:42 2023

This file contains a "toolbox" library - functions and constants that will be
used in other scripts to perform the tasks associated with TCBench

@author: mgomezd1
"""

#%% Imports and base paths

import os
import pandas as pd
import numpy as np
import constants

# Retrieve Repository Path
repo_path = '/'+os.path.join(*os.getcwd().split('/')[:-1])

print(repo_path)
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

# Class presenting each track type
class tc_track:
    def __init__(self,
                 UID,
                 NAME,
                 track,
                 timestamps):
        assert type(UID) == str, f'Unique Identifier Error. {type(UID)} given when string was expected'
        assert type(NAME) == str, f'Name Error. {type(NAME)} given when string was expected'
        assert type(track) == str, f'Track Error. {type(track)} given when numpy array was expected'
        assert (type(timestamps) == np.ndarray or type(timestamps) == pd.DataFrame), f'Unsupported timestamp iterator {type(timestamps)}'
        assert np.issubdtype(timestamps.dtype, np.datetime64), f'Unsupported timestamp type: {timestamps.dtype} - expected datetime64'
        
        self.uid = UID
        self.name = NAME
        self.track = track,
        self.timestamps = timestamps
        
    
    def get_varmask(self,
                    grid,
                    distance_calculator,
                    ):
        pass


#%% Auxilliary Functions

# Lat-lon grid generator
def ll_gridder(origin = (0,0),
               resolution = 0.25,
               lon_mode = 'pos',
               ):
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
    pass

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