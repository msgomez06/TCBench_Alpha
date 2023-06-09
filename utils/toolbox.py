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
import constants

# Retrieve Repository Path
repo_path = '/'+os.path.join(*os.getcwd().split('/')[:-1])

print(repo_path)
#%% Data Processing / Preprocessing Library
"""
This cell contains the following classes and functions:
    
"""

def read_py_track_file(tracks_path = f'{repo_path}/tracks/',
                       year = 2000,
                       backend = pd.read_csv
                       ):
    """
    Function to read per-year separated storm track files
    
    Parameters
    ----------
    tracks_path : STR, required
        Path to the location holding the track paths separated by year. 
        The default is f'{repo_path}/tracks/'. By default the files are
        expected to be in csv format
    year : INT, required
        year of the tracks to load. The default is 2000.
    backend : FUNC, required
        file handler for loading the track file. The default is pandas' 
        read_csv function.

    Returns
    -------
    track_object : type defined by backend. Default is pandas 

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
    
    pass

class tc_track:
    def __init__(self,
                 ):
        pass