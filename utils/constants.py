#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:53:01 2023

This library contains dictionaries and values useful as a reference for other
scripts used in TC Bench

@author: mgomezd1
"""

#%% Library Imports
import numpy as np

#%% Columns for track files

class track_cols:
    def __init__(self,
                 **kwargs):
        for key, val in kwargs.items():
            setattr(self, 
                    key, 
                    val)
        pass

# Track column data for tracks prepared by Dr. Saranya Ganesh Sudheesh
sgs_track_cols = track_cols(# Basin Corresponts to ibtracs basin
                            basin = {'NA':'North Atlantic',
                                     'EP':'Eastern North Pacific',
                                     'WP':'Western North Pacific',
                                     'NI':'North Indian',
                                     'SI':'South Indian',
                                     'SP':'South Pacific',
                                     'SA':'South Atlantic',},
                            
                            # Region Corresponts to ibtracs subbasin
                            region = {'MM':'Missing',
                                      'CS':'Caribbean Sea',
                                      'GM':'Gulf of Mexico',
                                      'CP':'Central Pacific',
                                      'BB':'Bay of Bengal',
                                      'AS':'Arabian Sea',
                                      'WA':'Western Australia',
                                      'EA':'Eastern Australia',},
                            
                            # Name of the storm
                            name = str,
                            
                            # Track step
                            no = int,
                            
                            # Track point timestamp
                            datetime = np.datetime64,
                            
                            # Intensity corresponds to ibtracs nature
                            intensity = {'DS':'Disturbance',
                                         'TS':'Tropical',
                                         'ET':'Extratropical',
                                         'SS':'Subtropical',
                                         'NR':'Not Reported',
                                         'MX':'Mixture',
                                         },
                            
                            # Coordinates
                            lat = np.float16,
                            lon = np.float16,
                        )

#%%
