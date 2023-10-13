#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:53:01 2023

This library contains dictionaries and values useful as a reference for other
scripts used in TC Bench

@author: mgomezd1
"""

# %% Library Imports
import numpy as np
import pandas as pd


# %% Columns for track files
class track_cols:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if key != "__META":
                setattr(self, key, val)
            else:
                self.__metadata = val

    def get_dtypes(self):
        dtype_dict = {}

        for key, dtype in self.__dict__.items():
            if key[0] != "_":
                dtype_dict[key] = (
                    str
                    if (isinstance(dtype, dict) or np.issubdtype(dtype, np.datetime64))
                    else dtype
                )
        return dtype_dict

    def get_colnames(self):
        colnames = []
        for key, dtype in self.__dict__.items():
            if key[0] != "_":
                colnames.append(key)
        return colnames

    def get_datetime_cols(self):
        datetime_cols = []

        for key, dtype in self.__dict__.items():
            if dtype == np.datetime64:
                datetime_cols.append(key)
        return datetime_cols


# Track column data for tracks prepared by Dr. Saranya Ganesh Sudheesh
sgs_track_cols = track_cols(  # Basin Corresponts to ibtracs basin
    basin={
        "NA": "North Atlantic",
        "EP": "Eastern North Pacific",
        "WP": "Western North Pacific",
        "NI": "North Indian",
        "SI": "South Indian",
        "SP": "South Pacific",
        "SA": "South Atlantic",
    },
    # Region Corresponts to ibtracs subbasin
    region={
        "MM": "Missing",
        "CS": "Caribbean Sea",
        "GM": "Gulf of Mexico",
        "CP": "Central Pacific",
        "BB": "Bay of Bengal",
        "AS": "Arabian Sea",
        "WA": "Western Australia",
        "EA": "Eastern Australia",
    },
    # Name of the storm
    name=str,
    # Track step
    no=int,
    # Track point timestamp
    datetime=np.datetime64,
    # Intensity corresponds to ibtracs nature
    intensity={
        "DS": "Disturbance",
        "TS": "Tropical",
        "ET": "Extratropical",
        "SS": "Subtropical",
        "NR": "Not Reported",
        "MX": "Mixture",
    },
    # Coordinates
    lat=np.float16,
    lon=np.float16,
    # Metadata for constructing track objects
    __META={
        "UID": "SID",
        "COSMETIC_NAME": "NAME",
        "Y_coord": "lat",
        "X_coord": "lon",
        "TIME_coord": "datetime",
        "loader": pd.read_csv,
    },
)

# Track column data for ibtracs file
ibtracs_cols = track_cols(  # Storm ID number
    SID=str,
    # Year of the event
    SEASON=int,
    # Number of event in season CURRENTLY IGNORED
    # NUMBER = int
    # Basin
    BASIN={
        "NA": "North Atlantic",
        "EP": "Eastern North Pacific",
        "WP": "Western North Pacific",
        "NI": "North Indian",
        "SI": "South Indian",
        "SP": "South Pacific",
        "SA": "South Atlantic",
        "MM": "Missing",  # Note that MM should not appear in final ibtracs product
    },
    # Subbasin
    SUBBASIN={
        "MM": "Missing",
        "CS": "Caribbean Sea",
        "GM": "Gulf of Mexico",
        "CP": "Central Pacific",
        "BB": "Bay of Bengal",
        "AS": "Arabian Sea",
        "WA": "Western Australia",
        "EA": "Eastern Australia",
    },
    # Name of the storm provided by the agencies
    NAME=str,
    # Track point ISO timestamp
    ISO_TIME=np.datetime64,
    # IBTRACS Combined storm type
    NATURE={
        "DS": "Disturbance",
        "TS": "Tropical",
        "ET": "Extratropical",
        "SS": "Subtropical",
        "NR": "Not Reported",
        "MX": "Mixture",
    },
    # Coordinates
    LAT=np.float16,
    LON=np.float16,
    # US Automated Tropical Cyclone Forecasting System ID
    USA_ATCF_ID=str,
    # Intensity and Pressure
    WMO_WIND=float,
    WMO_PRES=float,
    # Metadata for constructing track objects
    __META={
        "UID": "SID",
        "COSMETIC_NAME": "NAME",
        "Y_coord": "LAT",
        "X_coord": "LON",
        "TIME_coord": "ISO_TIME",
        "loader": pd.read_csv,
    },
)

# Track column data for Meteo France La Reunion file
reunion_track_cols = track_cols(  # Storm Season
    SAISON=int,
    # Name of the storm
    NOM_CYC=str,
    # Seasonal number of the depression
    NUM_DEPR=int,
    # Storm ID
    ID=str,
    # Track point timestamp
    ANNEE=np.datetime64,
    MOIS=np.datetime64,
    JOUR=np.datetime64,
    HEURE_UTC=np.datetime64,
    # Coordinates
    LAT=np.float16,
    LON=np.float16,
    # Metadata for constructing track objects
    __META={
        "UID": "ID",
        "COSMETIC_NAME": "NOM_CYC",
        "Y_coord": "LAT",
        "X_coord": "LON",
        "TIME_coord": "date",
        "loader": "meteo_france",
    },
)

# %% PRIMED Metadata
# Questions for developers:
# Why are the files organized by these basins but the metadata includes
# basins? Why are the basins not the same?
primed_basins = set(
    [
        "AL",  # North Atlantic
        "CP",  # Central Pacific
        "EP",  # Eastern North Pacific
        "IO",  # Indian Ocean
        "SH",  # Southern Hemisphere
        "WP",  # Western North Pacific
    ]
)

# Primed storm names are stored in a nested dictionary, the first key dedicated
# to the year and the second key dedicated to the storm ID. Note that this has
# currently only been applied to the North Atlantic basin. For storms without a name,
# the dictionary return by the year should be called using the .get() method.
# e.g., psn[study_year].get(STORM_ID, 'Unnamed')
# Primed Storm Names abbreviated to psn
psn = {
    2000: {
        3: "Alberto",
        10: "Florence",
        11: "Gordon",
        12: "Helene",
        13: "Isaac",
        14: "Joyce",
        15: "Keith",
        16: "Leslie",
    },
    2001: {
        1: "Allison",
        3: "Barry",
        4: "Chantal",
        5: "Dean",
        6: "Erin",
        7: "Felix",
        8: "Gabrielle",
        10: "Humberto",
        11: "Iris",
        15: "Michelle",
        17: "Olga",
    },
    2002: {4: "Dolly", 8: "Gustav", 10: "Isidore", 12: "Kyle", 13: "Lili"},
    2003: {
        1: "Ana",
        4: "Claudette",
        5: "Danny",
        10: "Fabian",
        13: "Isabel",
        16: "Kate",
        17: "Larry",
        19: "Nicholas",
    },
    2004: {
        1: "Ana",
        2: "Bonnie",
        3: "Charley",
        4: "Danielle",
        6: "Frances",
        7: "Gaston",
        9: "Ivan",
        11: "Jeanne",
        12: "Karl",
        13: "Lisa",
        16: "Otto",
    },
    2005: {
        1: "Arlene",
        3: "Cindy",
        4: "Dennis",
        5: "Emily",
        6: "Franklin",
        8: "Harvey",
        9: "Irene",
        12: "Katrina",
        13: "Lee",
        14: "Maria",
        15: "Nate",
        16: "Ophelia",
        17: "Philippe",
        18: "Rita",
        25: "Wilma",
        28: "Gamma",
        29: "Delta",
        30: "Epsilon",
        31: "Zeta",
    },
    2006: {
        1: "Alberto",
        5: "Debby",
        6: "Ernesto",
        7: "Florence",
        8: "Gordon",
        9: "Helene",
    },
    2007: {
        1: "Andrea",
        4: "Dean",
        6: "Felix",
        8: "Ingrid",
        11: "Jerry",
        12: "Karen",
        14: "Melissa",
        16: "Noel",
        17: "Olga",
    },
    2008: {
        # 1: "Arthur",
        2: "Bertha",
        # 3: "Cristobal",
        4: "Dolly",
        # 5: "Edouard",
        6: "Fay",
        7: "Gustav",
        8: "Hanna",
        9: "Ike",
        10: "Josephine",
        # 11: "Kyle",
        12: "Laura",
        # 13: "Marco",
        # 14: "Nana",
        15: "Omar",
        17: "Paloma",
    },
    2009: {
        2: "Ana",
        3: "Bill",
        # 4:'Claudette',
        # 5:'Danny',
        # 6:'Erika',
        7: "Fred",
        9: "Grace",
        # 10:'Henri',
        11: "Ida",
    },
}
