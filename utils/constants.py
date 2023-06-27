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
import pandas as pd




#%% Columns for track files
class track_cols:
    def __init__(self,
                 **kwargs):
        for key, val in kwargs.items():
            
            if key !='__META':
                setattr(self, 
                        key, 
                        val)
            else:
                self.__metadata = val
    
    def get_dtypes(self):
        dtype_dict = {}
        
        for key, dtype in self.__dict__.items():
            if key[0] != '_':
                dtype_dict[key] = dtype if (type(dtype) != dict and dtype!=np.datetime64) else str
        return dtype_dict
   
    def get_colnames(self):
        colnames = []
        for key, dtype in self.__dict__.items():
            if key[0]!='_': 
                colnames.append(key)
        return colnames
   
    def get_datetime_cols(self):
        
        datetime_cols = []
        
        for key, dtype in self.__dict__.items():
            if dtype==np.datetime64: datetime_cols.append(key)
        return datetime_cols

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
                        
                            # Metadata for constructing track objects
                            __META = {'UID':'SID',
                                      'COSMETIC_NAME':'NAME',
                                      'Y_coord':'lat',
                                      'X_coord':'lon',
                                      'TIME_coord':'datetime',
                                      'loader':pd.read_csv,}
                            )

# Track column data for ibtracs file
ibtracs_cols = track_cols(# Storm ID number
                          SID = str,
                          
                          # Year of the event
                          SEASON = int,
                          
                          # Number of event in season CURRENTLY IGNORED
                          # NUMBER = int
                          
                          # Basin
                          BASIN = {'NA':'North Atlantic',
                                   'EP':'Eastern North Pacific',
                                   'WP':'Western North Pacific',
                                   'NI':'North Indian',
                                   'SI':'South Indian',
                                   'SP':'South Pacific',
                                   'SA':'South Atlantic',
                                   'MM':'Missing' #Note that MM should not appear in final ibtracs product
                                   },
                            
                          # Subbasin
                          SUBBASIN = {'MM':'Missing',
                                      'CS':'Caribbean Sea',
                                      'GM':'Gulf of Mexico',
                                      'CP':'Central Pacific',
                                      'BB':'Bay of Bengal',
                                      'AS':'Arabian Sea',
                                      'WA':'Western Australia',
                                      'EA':'Eastern Australia',
                                      },
                            
                          # Name of the storm provided by the agencies
                          NAME = str,
                            
                          # Track point ISO timestamp
                          ISO_TIME = np.datetime64,
                            
                          # IBTRACS Combined storm type
                          NATURE = {'DS':'Disturbance',
                                    'TS':'Tropical',
                                    'ET':'Extratropical',
                                    'SS':'Subtropical',
                                    'NR':'Not Reported',
                                    'MX':'Mixture',
                                    },
                            
                          # Coordinates
                          LAT = np.float16,
                          LON = np.float16,
                          
                          # Metadata for constructing track objects
                          __META = {'UID':'SID',
                                    'COSMETIC_NAME':'NAME',
                                    'Y_coord':'LAT',
                                    'X_coord':'LON',
                                    'TIME_coord':'ISO_TIME',
                                    'loader':pd.read_csv,}
                          )

# Track column data for Meteo France La Reunion file
reunion_track_cols = track_cols(# Storm Season
                                SAISON = int,
                                
                                # Name of the storm
                                NOM_CYC = str,
                                
                                # Seasonal number of the depression
                                NUM_DEPR = int,
                                
                                # Storm ID
                                ID = str,

                                # Track point timestamp
                                ANNEE = np.datetime64,
                                MOIS = np.datetime64,
                                JOUR = np.datetime64,
                                HEURE_UTC = np.datetime64,

                                
                                # Coordinates
                                LAT = np.float16,
                                LON = np.float16,
                            
                                # Metadata for constructing track objects
                                __META = {'UID':'ID',
                                        'COSMETIC_NAME':'NOM_CYC',
                                        'Y_coord':'LAT',
                                        'X_coord':'LON',
                                        'TIME_coord':'date',
                                        'loader':'meteo_france',}
                                )

#%%
