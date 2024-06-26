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
    USA_WIND=str,
    USA_PRES=str,
    # Metadata for constructing track objects
    __META={
        "UID": "SID",
        "COSMETIC_NAME": "NAME",
        "Y_coord": "LAT",
        "X_coord": "LON",
        "TIME_coord": "ISO_TIME",
        "loader": pd.read_csv,
        "ALT_ID": "USA_ATCF_ID",
        "WIND": "USA_WIND",
        "PRES": "USA_PRES",
        "SEASON": "SEASON",
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

# %% Folder Structure Long Names
# The folder dictionary is used to expand the folder names to their full
# names. This is useful for the user to understand what the folder names
# mean.
# The ERA5 folder structure is organized by single level, pressure level, and
# calculated values.
data_store_names = {
    "SL": "Surface / Single Level",
    "PL": "Pressure Level",
    "CV": "Calculated Values",
}

# %% valid coordinate names
valid_coords = {
    "latitude": ["lat", "Lat"],
    "longitude": ["lon", "Lon"],
    "time": ["time", "Time", "t"],
    "level": ["Lev", "lev", "pressure", "Pressure", "isobaric"],
    "leadtime": ["lead", "Lead", "ldt", "LDT", "valid", "Valid"],
}

# default variable names
default_ships_vars = [
    "DELV",  # Intensity change
    "VMAX",  # Maximum sustained wind
    "MSLP",  # Minimum sea level pressure
    "lat",  # Latitude
    "CSST",  # Climatological sea surface temperature
    # "PSLV",  # Climatological sea level pressure
    "Z850",  # 850 hPa geopotential height
    "D200",  # 200 hPa divergence
    "EPOS",  # Surface - Environment theta_e difference, r = 200-800
    "RHMD",  # 700-500 hPa relative humidity
    "TWAC",  # Average symmetric tangential wind at 850 hPa (0-600km average)
    "G200",  # Averaged 200 hPa Temperature Perturbation (r=200-800km)
    "TADV",  # Averaged 850 to 700 hPa temperature advection, r=0-500km
    "SHGC",  # Generalized 850-200 hPa shear magnitude, vortex removed (r=0-500km)
    "LHRD",  # SHDC * sin(lat)
    "VSHR",  # VMAX*SHDC
    "T200",  # 200 hPa temperature
    "T250",  # 250 hPa temperature
    "SHDC",  # 850-200 hPa shear magnitude, vortex removed (r=0-500km)
]

# %% SHIPS Metadata
SHIPS_metadata = {
    "VMAX": {"long_name": "Maximum Surface Wind", "units": "kt"},
    "MSLP": {"long_name": "Minimum Sea Level Pressure", "units": "hPa"},
    "TYPE": {
        "long_name": "Storm type",
        "units": "None",
        "Notes": "0=wave, remnant low, dissipating low, 1=tropical, 2=subtropical, 3=extra-tropical. \n Note that the SHIPS variables are set to missing for all cases except type=1 or 2,\nsince these are not included in the SHIPS developmental sample for estimating the model coefficients.",
    },
    "HIST": {
        "long_name": "Storm history",
        "units": "6hr periods",
        "Notes": "The no. of 6 hour periods the storm max wind has been above 20, 25, ,…, 120 kt.",
        "LeadTimeVar": False,
    },
    "DELV": {
        "long_name": "Intensity change (see notes)",
        "units": "kt",
        "Notes": "-12 to 0, -6 to 0, 0 to 0, 0 to 6, ... 0 to 120 hr. If the storm crosses a major land mass during the time interval, value set to NaN (9999 in original dataset)",
        "LeadTimeVar": False,
    },
    "INCV": {
        "long_name": "Intensity change (see notes)",
        "units": "kt",
        "Notes": "-18 to -12, -12 to -6, ... 114 to 120 hr. Set to 9999 similar to DELV for land cases.",
        "LeadTimeVar": False,
    },
    "LAT": {"long_name": "Storm latitude", "units": "deg N *10", "Notes": "vs time"},
    "LON": {"long_name": "Storm longitude", "units": "deg W *10", "Notes": "vs time"},
    "CSST": {
        "long_name": "Climatological SST",
        "units": "deg C * 10",
        "Notes": "vs time",
    },
    "CD20": {
        "long_name": "Climatological depth of 20 deg C isotherm",
        "units": "m",
        "Notes": "from 2005-2010 NCODA analyses",
    },
    "CD26": {
        "long_name": "Climatological depth of 26 deg C isotherm",
        "units": "m",
        "Notes": "from 2005-2010 NCODA analyses",
    },
    "COHC": {
        "long_name": "Climatological ocean heat content",
        "units": "kJ/cm2",
        "Notes": "from 2005-2010 NCODA analyses",
    },
    "DTL": {
        "long_name": "Distance to nearest major land mass",
        "units": "km",
        "Notes": "vs time",
    },
    "OAGE": {
        "long_name": "Ocean Age",
        "units": "hr*10",
        "Notes": "The amount of time the area within 100 km of the storm center has been occupied by the storm along its track up to this point in time",
    },
    "NAGE": {
        "long_name": "Normalized Ocean Age",
        "units": "hr*10",
        "Notes": "Same as OAGE, but normalized by the maximum wind/100kt. If the max wind was a constant 100 kt over its past history, NAGE=OAGE.",
    },
    "RSST": {
        "long_name": "Reynolds Sea Surface Temperature",
        "units": "deg C*10",
        "Notes": "vs time",
        "Info_Label": "Age, in days, of the SST analysis used to estimate RSST",
    },
    "DSST": {
        "long_name": "Daily Reynolds Sea Surface Temperature",
        "units": "deg C*10",
        "Notes": "vs time",
        "Info_Label": "Age, in days, of the SST analysis used to estimate RSST",
    },
    "DSTA": {
        "long_name": "Spatially Averaged Daily Reynolds Sea Surface Temperature",
        "units": "deg C*10",
        "Notes": "vs time. spatially averaged over 5 points (storm center, + 50 km N, E, S and W of center)",
        "Info_Label": "Age, in days, of the SST analysis used to estimate RSST",
    },
    "PHCN": {
        "long_name": "Estimated ocean heat content",
        "units": "kJ/cm2",
        "Notes": "From climo OHC and current SST anomaly. Designed to fill in for RHCN when that is missing.",
    },
    "U200": {"long_name": "200 hPa zonal wind, r=200-800 km", "units": "kt *10"},
    "U20C": {"long_name": "200 hPa zonal wind, r=0-500 km", "units": "kt *10"},
    "V20C": {"long_name": "200 hPa meridional wind, r=0-500 km", "units": "kt *10"},
    "E000": {
        "long_name": "1000 hPa Potential Temperature (Theta_e)",
        "units": "Unit",
        "Notes": "",
    },
    "EPOS": {
        "long_name": "Surface - Environment Potential Temperature (theta_e) difference, r = 200-800",
        "units": "deg K*10",
        "Notes": "The average theta e difference between a parcel lifted from the surface and its environment \n(200-800 km average) vs time. Only positive differences are included in the average.",
    },
    "ENEG": {
        "long_name": "Negative Surface - Environment Potential Temperature (theta_e) difference, r = 200-800",
        "units": "Unit",
        "Notes": "The average theta e difference between a parcel lifted from the surface and its environment \n(200-800 km average) vs time. Only negative differences are included in the average.\nMinus sign not included.",
    },
    "EPSS": {
        "long_name": "Negative Surface - Saturated-Environment Potential Temperature (theta_e) difference, r = 200-800",
        "units": "Unit",
        "Notes": "The average theta_e difference between a parcel lifted from the surface and the saturated theta_e \nof its environment (200-800 km average) vs time. Only positive differences are included in the average.",
    },
    "ENSS": {
        "long_name": "Negative Surface - Saturated-Environment Potential Temperature (theta_e) difference, r = 200-800",
        "units": "Unit",
        "Notes": "The average theta_e difference between a parcel lifted from the surface and the saturated theta_e \nof its environment (200-800 km average) vs time. Only negative differences are included in the average.\nMinus sign not included.",
    },
    "RHLO": {
        "long_name": "850-700 hPa relative humidity, r=200-800 km",
        "units": "%",
        "Notes": "vs time",
    },
    "RHMD": {
        "long_name": "700-500 hPa relative humidity, r=200-800 km",
        "units": "%",
        "Notes": "vs time",
    },
    "RHHI": {
        "long_name": "500-300 hPa relative humidity, r=200-800 km",
        "units": "%",
        "Notes": "vs time",
    },
    "PSLV": {
        "long_name": "Pressure, Storm Center Motion, Deep Layer Wind, and Weights",
        "units": "See Notes",
        "Notes": f"""
                 First Column: Pressure of the center of mass of the layer where storm motion best matches environmental flow (hPa) (t=0 only)
                 Second Column: The observed zonal storm motion component (m/s *10)
                 Third Column: The observed meridional storm motion component (m/s *10)
                 Fourth Column: The observed zonal storm motion component (m/s *10) for the 1000 to 100 hPa mass weighted deep layer environmental wind
                 Fifth Column: The observed meridional storm motion component (m/s *10) for the 1000 to 100 hPa mass weighted deep layer environmental wind
                 Sixth Column: The observed zonal storm motion component (m/s *10) for the optimally weighted deep layer mean flow
                 Seventh Column: The observed meridional storm motion component (m/s *10) for the optimally weighted deep layer mean flow
                 Eighth Column: The parameter alpha that controls the constraint on the weights from being not too “far” from the deep layer mean weights (non-dimensional, *100)
                 Ninth Column: The optimal vertical weights for p=100 hPa. (non-dimensional *1000)
                 Tenth Column: The optimal vertical weights for p=150 hPa. (non-dimensional *1000)
                 Eleventh Column: The optimal vertical weights for p=200 hPa. (non-dimensional *1000)
                 Twelfth Column: The optimal vertical weights for p=250 hPa. (non-dimensional *1000)
                 Thirteenth Column: The optimal vertical weights for p=300 hPa. (non-dimensional *1000)
                 Fourteenth Column: The optimal vertical weights for p=400 hPa. (non-dimensional *1000)
                 Fifteenth Column: The optimal vertical weights for p=500 hPa. (non-dimensional *1000)
                 Sixteenth Column: The optimal vertical weights for p=700 hPa. (non-dimensional *1000)
                 Seventeenth Column: The optimal vertical weights for p=850 hPa. (non-dimensional *1000)
                 Eighteenth Column: The optimal vertical weights for p=1000 hPa. (non-dimensional *1000)""",
        "LeadTimeVar": False,
        "Data_Length": 18,
    },
    "Z850": {
        "long_name": "850 hPa vorticity (r=0-1000km)",
        "units": "sec-1 * 10**7",
        "Notes": "vs time",
    },
    "D200": {
        "long_name": "200 hPa divergence (r=0-1000km)",
        "units": "sec-1 * 10**7",
        "Notes": "vs time",
    },
    "REFC": {
        "long_name": "Relative eddy momentum flux convergence, 100-600 km avg",
        "units": "m/sec/day",
        "Notes": "vs time",
    },
    "PEFC": {
        "long_name": "Planetary eddy momentum flux convergence, 100-600 km avg",
        "units": "m/sec/day",
        "Notes": "vs time",
    },
    "T000": {
        "long_name": "1000 hPa temperature, 200-800 km average",
        "units": "dec C* 10",
        "Notes": "",
    },
    "R000": {
        "long_name": "1000 hPa relative humidity, 200-800 km average",
        "units": "%",
        "Notes": "",
    },
    "Z000": {
        "long_name": "1000 hPa height deviation",
        "units": "m",
        "Notes": "Height deviation from the U.S. standard atmosphere",
    },
    "TLAT": {
        "long_name": "850 hPa vortex center latitude",
        "units": "deg N*10",
        "Notes": "Vortex in the NCEP analysis",
    },
    "TLON": {
        "long_name": "850 hPa vortex center longitude",
        "units": "deg W*10",
        "Notes": "Vortex in the NCEP analysis",
    },
    "TWAC": {
        "long_name": "Average symmetric tangential wind at 850 hPa (0-600km average)",
        "units": "m/sec *10",
        "Notes": "from NCEP analysis",
    },
    "TWXC": {
        "long_name": "Maximum 850 hPa symmetric tangential wind",
        "units": "m/sec *10",
        "Notes": "from NCEP analysis",
    },
    "G150": {
        "long_name": "Averaged 150 hPa Temperature Perturbation (r=200-800km)",
        "units": "deg C*10",
        "Notes": """Perturbation due to the symmetric vortex calculated from the gradient thermal wind.
        Radius centered on input lat/lon (not always the model/analysis vortex position)""",
    },
    "G200": {
        "long_name": "Averaged 200 hPa Temperature Perturbation (r=200-800km)",
        "units": "deg C*10",
        "Notes": """Perturbation due to the symmetric vortex calculated from the gradient thermal wind.
        Radius centered on input lat/lon (not always the model/analysis vortex position)""",
    },
    "G250": {
        "long_name": "Averaged 250 hPa Temperature Perturbation (r=200-800km)",
        "units": "deg C*10",
        "Notes": """Perturbation due to the symmetric vortex calculated from the gradient thermal wind.
        Radius centered on input lat/lon (not always the model/analysis vortex position)""",
    },
    "V000": {
        "long_name": "Azimuthally averaged 1000 hPa tangential wind (r=500km)",
        "units": "m/sec *10",
        "Notes": "Azimuthally averaged at r=500 km from (TLAT,TLON). \nIf TLAT,TLON are not available, (LAT,LON) are used.",
    },
    "V850": {
        "long_name": "Azimuthally averaged 850 hPa tangential wind (r=500km)",
        "units": "m/sec *10",
        "Notes": "Azimuthally averaged at r=500 km from (TLAT,TLON). \nIf TLAT,TLON are not available, (LAT,LON) are used.",
    },
    "V500": {
        "long_name": "Azimuthally averaged 500 hPa tangential wind (r=500km)",
        "units": "m/sec *10",
        "Notes": "Azimuthally averaged at r=500 km from (TLAT,TLON). \nIf TLAT,TLON are not available, (LAT,LON) are used.",
    },
    "V300": {
        "long_name": "Azimuthally averaged 300 hPa tangential wind (r=500km)",
        "units": "m/sec *10",
        "Notes": "Azimuthally averaged at r=500 km from (TLAT,TLON). \nIf TLAT,TLON are not available, (LAT,LON) are used.",
    },
    "TGRD": {
        "long_name": "Averaged 850 to 700 hPa temperature gradient magnitude, r=0-500km",
        "units": "deg C per m*107",  # Is this right? Verify with @Mark and @Kate
        "Notes": "Estimated from the geostrophic thermal wind",
    },
    "TADV": {
        "long_name": "Averaged 850 to 700 hPa temperature advection, r=0-500km",
        "units": "deg per sec*106",  # Is this right? Verify with @Mark and @Kate
        "Notes": "Estimated from the geostrophic thermal wind",
    },
    "PENC": {
        "long_name": "Azimuthally averaged surface pressure at outer edge of vortex",
        "units": "(hPa-1000)*10",
        "Notes": "",
    },
    "SHRD": {
        "long_name": "850-200 hPa shear magnitude (200-800km)",
        "units": "kt *10",
        "Notes": "vs time",
    },
    "SHDC": {
        "long_name": "850-200 hPa shear magnitude, vortex removed (r=0-500km)",
        "units": "kt *10",
        "Notes": "vs time. Averaged from 0-500 km relative to 850 hPa vortex center",
    },
    "SDDC": {
        "long_name": "Heading of above shear vector",
        "units": "deg",
        "Notes": "Westerly shear has a value of 90 deg",
    },
    "SHRG": {
        "long_name": "Generalized 850-200 hPa shear magnitude",
        "units": "kt *10",
        "Notes": "vs time. Takes into account all levels from 1000 to 100 hPa",
    },
    "SHGC": {
        "long_name": "Generalized 850-200 hPa shear magnitude, vortex removed (r=0-500km)",
        "units": "kt *10",
        "Notes": "vs time. Takes into account all levels from 1000 to 100 hPa. \nAveraged from 0-500 km relative to 850 hPa vortex center",
    },
    "DIVC": {
        "long_name": "Centered 200 hPa divergence (r=0-1000km)",
        "units": "sec-1 * 10**7",
        "Notes": "vs time. Centered at 850 hPa vortex location",
    },
    "T150": {
        "long_name": "200 to 800 km area average 150 hPa temperature",
        "units": "deg C *10",
        "Notes": "versus time",
    },
    "T200": {
        "long_name": "200 to 800 km area average 200 hPa temperature",
        "units": "deg C *10",
        "Notes": "versus time",
    },
    "T250": {
        "long_name": "200 to 800 km area average 250 hPa temperature",
        "units": "deg C *10",
        "Notes": "versus time",
    },
    "SHTD": {  # How is this different from SHTS? SDDC? @Mark and @Kate
        "long_name": "Heading of above shear vector",
        "units": "deg",
        "Notes": "Westerly shear has a value of 90 deg.",
    },
    "SHRS": {
        "long_name": "850-500 hPa shear magnitude",
        "units": "kt *10",
        "Notes": "vs time",
    },
    "SHTS": {
        "long_name": "Heading of above shear vector",
        "units": "deg",  # verify with @Mark and @Kate
        "Notes": "",  # Should this say 90deg for Westerly shear? verify with @Mark and @Kate
    },
    "PENV": {
        "long_name": "200 to 800 km average surface pressure",
        "units": "(hPa-1000)*10",
        "Notes": "",
    },
    "VMPI": {
        "long_name": "Maximum potential intensity",
        "units": "kt",
        "Notes": "from Kerry Emanuel equation",
    },
    "VVAV": {
        "long_name": "Average surface parcel vertical velocity",
        "units": "m/s * 100",
        "Notes": f"""Average vertical velocity (0 to 15 km) of a parcel lifted from the surface
                 where entrainment, the ice phase and the condensate weight are accounted for.
                 Note: Moisture and temperature biases between the operational and reanalysis files
                 make this variable inconsistent in the 2001-2007 sample, compared 2000 and before.""",
    },
    "VMFX": {
        "long_name": "Density Weighted Average surface parcel vertical velocity",
        "units": "m/s * 100",
        "Notes": f"""Average vertical velocity (0 to 15 km) of a parcel lifted from the surface
                 where entrainment, the ice phase and the condensate weight are accounted for.
                 Density weighted vertical average.
                 Note: Moisture and temperature biases between the operational and reanalysis files
                 make this variable inconsistent in the 2001-2007 sample, compared 2000 and before.""",
    },
    "VVAC": {
        "long_name": "Average surface parcel vertical velocity, vortex removed, from soundings",
        "units": "m/s * 100",
        "Notes": f"""
                 Same as VVAV but with soundings from 0-500 km with GFS vortex removed
                 Average vertical velocity (0 to 15 km) of a parcel lifted from the surface
                 where entrainment, the ice phase and the condensate weight are accounted for.
                 Note: Moisture and temperature biases between the operational and reanalysis files
                 make this variable inconsistent in the 2001-2007 sample, compared 2000 and before.""",
    },
    "HE07": {
        "long_name": "1000-700 hPa Storm motion relative helicity, r=200-800 km",
        "units": "m^2/s^2",
        "Notes": "",
    },
    "HE05": {
        "long_name": "1000-500 hPa Storm motion relative helicity, r=200-800 km",
        "units": "m^2/s^2",
        "Notes": "",
    },
    "O500": {
        "long_name": "Average 500 hPa Pressure vertical velocity, r=0-1000 km ",
        "units": "(hPa/day",
        "Notes": "",
    },
    "O700": {
        "long_name": "Average 700 hPa Pressure vertical velocity, r=0-1000 km ",
        "units": "(hPa/day",
        "Notes": "",
    },
    "CFLX": {
        "long_name": "Dry air predictor",
        "units": "%",  # veriy with @Mark and @ Kate
        "Notes": "based on the difference in surface moisture flux between air with the \nobserved (GFS) RH value,and with RH of air mixed \nfrom 500 hPa to the surface.",
    },
    "MTPW": {
        "long_name": "Total Precipitable Water (TPW) predictors @ t=0",
        "units": "See Notes",
        "Notes": f"""
                     1)	0-200 km average TPW (mm * 10)
                     2)	0-200 km TPW standard deviation (mm * 10)
                     3)	200-400 km average TPW (mm * 10)
                     4)	200-400 km TPW standard deviation (mm * 10)
                     5)	400-600 km average TPW (mm * 10)
                     6)	400-600 km TPW standard deviation (mm * 10)
                     7)	600-800 km average TPW (mm * 10)
                     8)	600-800 km TPW standard deviation (mm * 10)
                     9)	800-1000 km average TPW (mm * 10)
                     10)	800-1000 km TPW standard deviation (mm * 10)
                     11)	0-400 km average TPW (mm * 10)
                     12)	0-400 km TPW standard deviation (mm * 10)
                     13)	0-600 km average TPW (mm * 10)
                     14)	0-600 km TPW standard deviation (mm * 10)
                     15)	0-800 km average TPW (mm * 10)
                     16)	0-800 km TPW standard deviation (mm * 10)
                     17)	0-1000 km average TPW (mm * 10)
                     18)	0-1000 km TPW standard deviation (mm * 10)
                     19)	%TPW less than 45 mm, r=0 to 500 km in 90 deg azimuthal quadrant centered on up-shear direction
                     20)	0-500 km averaged TPW (mm * 10) in 90 deg up-shear quadrant
                     21)	0-500 km average TPW (mm * 10)""",
        "LeadTimeVar": False,
        "Data_Length": 21,
    },
    "PW01": {
        "long_name": "0-200 km average TPW",
        "units": "mm*10",
        "Notes": "Time dependent",
    },
    "PW02": {
        "long_name": "0-200 km TPW standard deviation",
        "units": "mm*10",
        "Notes": "Time dependent",
    },
    "PW03": {
        "long_name": "200-400 km average TPW",
        "units": "mm*10",
        "Notes": "Time dependent",
    },
    "PW04": {
        "long_name": "200-400 km TPW standard deviation",
        "units": "mm*10",
        "Notes": "Time dependent",
    },
    "PW05": {
        "long_name": "400-600 km average TPW",
        "units": "mm*10",
        "Notes": "Time dependent",
    },
    "PW06": {
        "long_name": "400-600 km TPW standard deviation",
        "units": "mm*10",
        "Notes": "Time dependent",
    },
    "PW07": {
        "long_name": "600-800 km average TPW",
        "units": "mm*10",
        "Notes": "Time dependent",
    },
    "PW08": {
        "long_name": "600-800 km TPW standard deviation",
        "units": "mm*10",
        "Notes": "Time dependent",
    },
    "PW09": {
        "long_name": "800-1000 km average TPW",
        "units": "mm*10",
        "Notes": "Time dependent",
    },
    "PW10": {
        "long_name": "800-1000 km TPW standard deviation",
        "units": "mm*10",
        "Notes": "Time dependent",
    },
    "PW11": {"long_name": "0-400 km average TPW", "units": "mm*10", "Notes": ""},
    "PW12": {
        "long_name": "0-400 km TPW standard deviation",
        "units": "mm*10",
        "Notes": "Time dependent",
    },
    "PW13": {
        "long_name": "0-600 km average TPW",
        "units": "mm*10",
        "Notes": "Time dependent",
    },
    "PW14": {
        "long_name": "0-600 km TPW standard deviation",
        "units": "mm*10",
        "Notes": "Time dependent",
    },
    "PW15": {"long_name": "0-800 km average TPW", "units": "mm*10", "Notes": ""},
    "PW16": {
        "long_name": "0-800 km TPW standard deviation",
        "units": "mm*10",
        "Notes": "Time dependent",
    },
    "PW17": {"long_name": "0-1000 km average TPW", "units": "mm*10", "Notes": ""},
    "PW18": {
        "long_name": "0-1000 km TPW standard deviation",
        "units": "mm*10",
        "Notes": "Time dependent",
    },
    "PW19": {
        "long_name": "%TPW less than 45 mm",
        "units": "Unit",
        "Notes": "r=0 to 500 km in 90 deg azimuthal quadrant centered on up-shear direction. Time dependent",
    },
    "PW20": {
        "long_name": "0-500 km averaged TPW ",
        "units": "mm*10",
        "Notes": "in 90 deg up-shear quadrant. Time dependent",
    },
    "PW21": {
        "long_name": "0-500 km average TPW",
        "units": "mm*10",
        "Notes": "Time dependent",
    },
    "IR00": {
        "long_name": "Predictors from GOES data",
        "units": "See Notes",
        "Notes": f""" Not time dependent. The 20 values in this record are as follows:
                    1) Time (hhmm) of the GOES image, relative to this case
                    2) Average GOES ch 4 brightness temp (deg C *10), r=0-200 km
                    3) Stan. Dev. of GOES BT (deg C*10), r=0-200 km
                    4) Same as 2) for r=100-300 km
                    5) Same as 3) for r=100-300 km
                    6) Percent area r=50-200 km of GOES ch 4 BT < -10 C
                    7) Same as 6 for BT < -20 C
                    8) Same as 6 for BT < -30 C
                    9) Same as 6 for BT < -40 C
                    10) Same as 6 for BT < -50 C
                    11) Same as 6 for BT < -60 C
                    12) max BT from 0 to 30 km radius (deg C*10)
                    13) avg BT from 0 to 30 km radius (deg C*10)
                    14) radius of max BT (km)
                    15) min BT from 20 to 120 km radius (deg C*10)
                    16) avg BT from 20 to 120 km radius (deg C*10)
                    17)  radius of min BT (km)
                    18-20) Variables need for storm size estimation""",
        "LeadTimeVar": False,
        "Data_Length": 20,
    },
    "IRXX": {
        "long_name": "Inferred Predictors substituting GOES data",
        "units": "See Notes",
        "Notes": f"""Not time dependent.
                     The same as IR00 but generated from other predictors (not satellite data).
                     These should only be used to fill in for missing IR00 if needed.
                    1) Time (hhmm) of the GOES image, relative to this case
                    2) Average GOES ch 4 brightness temp (deg C *10), r=0-200 km
                    3) Stan. Dev. of GOES BT (deg C*10), r=0-200 km
                    4) Same as 2) for r=100-300 km
                    5) Same as 3) for r=100-300 km
                    6) Percent area r=50-200 km of GOES ch 4 BT < -10 C
                    7) Same as 6 for BT < -20 C
                    8) Same as 6 for BT < -30 C
                    9) Same as 6 for BT < -40 C
                    10) Same as 6 for BT < -50 C
                    11) Same as 6 for BT < -60 C
                    12) max BT from 0 to 30 km radius (deg C*10)
                    13) avg BT from 0 to 30 km radius (deg C*10)
                    14) radius of max BT (km)
                    15) min BT from 20 to 120 km radius (deg C*10)
                    16) avg BT from 20 to 120 km radius (deg C*10)
                    17)  radius of min BT (km)
                    18-20) Variables need for storm size estimation""",
        "LeadTimeVar": False,
        "Data_Length": 20,
    },
    "IRM1": {
        "long_name": "Predictors from GOES data 1.5hrs before t0",
        "units": "See Notes",
        "Notes": f""" Not time dependent. The 20 values in this record are as follows:
                    1) Time (hhmm) of the GOES image, relative to this case
                    2) Average GOES ch 4 brightness temp (deg C *10), r=0-200 km
                    3) Stan. Dev. of GOES BT (deg C*10), r=0-200 km
                    4) Same as 2) for r=100-300 km
                    5) Same as 3) for r=100-300 km
                    6) Percent area r=50-200 km of GOES ch 4 BT < -10 C
                    7) Same as 6 for BT < -20 C
                    8) Same as 6 for BT < -30 C
                    9) Same as 6 for BT < -40 C
                    10) Same as 6 for BT < -50 C
                    11) Same as 6 for BT < -60 C
                    12) max BT from 0 to 30 km radius (deg C*10)
                    13) avg BT from 0 to 30 km radius (deg C*10)
                    14) radius of max BT (km)
                    15) min BT from 20 to 120 km radius (deg C*10)
                    16) avg BT from 20 to 120 km radius (deg C*10)
                    17)  radius of min BT (km)
                    18-20) Variables need for storm size estimation""",
        "LeadTimeVar": False,
        "Data_Length": 20,
    },
    "IRM3": {
        "long_name": "Predictors from GOES data 3hrs before t0",
        "units": "See Notes",
        "Notes": f""" Not time dependent. The 20 values in this record are as follows:
                    1) Time (hhmm) of the GOES image, relative to this case
                    2) Average GOES ch 4 brightness temp (deg C *10), r=0-200 km
                    3) Stan. Dev. of GOES BT (deg C*10), r=0-200 km
                    4) Same as 2) for r=100-300 km
                    5) Same as 3) for r=100-300 km
                    6) Percent area r=50-200 km of GOES ch 4 BT < -10 C
                    7) Same as 6 for BT < -20 C
                    8) Same as 6 for BT < -30 C
                    9) Same as 6 for BT < -40 C
                    10) Same as 6 for BT < -50 C
                    11) Same as 6 for BT < -60 C
                    12) max BT from 0 to 30 km radius (deg C*10)
                    13) avg BT from 0 to 30 km radius (deg C*10)
                    14) radius of max BT (km)
                    15) min BT from 20 to 120 km radius (deg C*10)
                    16) avg BT from 20 to 120 km radius (deg C*10)
                    17)  radius of min BT (km)
                    18-20) Variables need for storm size estimation""",
        "LeadTimeVar": False,
        "Data_Length": 20,
    },
    "PC00": {
        "long_name": "IR Principal Components and related variables @t0",
        "units": "Unitless",  # confirm with @Mark and @Kate
        "Notes": "",
        "LeadTimeVar": False,
        "Data_Length": 21,  # confirm with @Mark and @Kate
    },
    "PCM1": {
        "long_name": "IR Principal Components and related variables @t0-1.5hrs",
        "units": "Unitless",  # confirm with @Mark and @Kate
        "Notes": "",
        "LeadTimeVar": False,
        "Data_Length": 21,  # confirm with @Mark and @Kate
    },
    "PCM3": {
        "long_name": "IR Principal Components and related variables @t0-3hrs",
        "units": "Unitless",  # confirm with @Mark and @Kate
        "Notes": "",
        "LeadTimeVar": False,
        "Data_Length": 21,  # confirm with @Mark and @Kate
    },
    "RD20": {
        "long_name": "Ocean depth of the 20 deg C isotherm",
        "units": "m",
        "Notes": "from satellite altimetry data",
    },
    "RD26": {
        "long_name": "Ocean depth of the 26 deg C isotherm",
        "units": "m",
        "Notes": "from satellite altimetry data",
    },
    "RHCN": {
        "long_name": "Ocean heat content",
        "units": "KJ/cm2",
        "Notes": "from satellite altimetry data",
        "Info_Label": "Age, in days, of the OHC analysis used to estimate RD20, RD26 and RHCN",
    },
    "NSST": {
        "long_name": "SST from the NCODA analysis",
        "units": "deg C*10",
        "Notes": "",
    },
    "NSTA": {
        "long_name": "Averaged SST from the NCODA analysis",
        "units": "deg C*10",
        "Notes": "spatially averaged over 5 points (storm center, + 50 km N, E, S and W of center)",
    },
    "NTMX": {
        "long_name": "Max ocean temperature in the NCODA vertical profile",
        "units": "deg C*10",
        "Notes": "",
    },
    "NDMX": {
        "long_name": "Depth of the max ocean temperature in the profile",
        "units": "m",
        "Notes": "",
    },
    "NDML": {
        "long_name": "Depth of the mixed layer",
        "units": "m",
        "Notes": """Defined as the depth where the T is 0.5 colder than at the surface. 
                    In rare cases, this is the depth where the T is 0.5 warmer than at the surface.
                    In those cases, NDML is negative.""",  # verify with @Mark and @Kate
    },
    "ND30": {
        "long_name": "Depth of the 30 deg C isotherm",
        "units": "m",
        "Notes": "",
    },
    "ND28": {
        "long_name": "Depth of the 28 deg C isotherm",
        "units": "m",
        "Notes": "",
    },
    "ND26": {
        "long_name": "Depth of the 26 deg C isotherm",
        "units": "m",
        "Notes": "",
    },
    "ND24": {
        "long_name": "Depth of the 24 deg C isotherm",
        "units": "m",
        "Notes": "",
    },
    "ND22": {
        "long_name": "Depth of the 22 deg C isotherm",
        "units": "m",
        "Notes": "",
    },
    "ND20": {
        "long_name": "Depth of the 20 deg C isotherm",
        "units": "m",
        "Notes": "",
    },
    "ND18": {
        "long_name": "Depth of the 18 deg C isotherm",
        "units": "m",
        "Notes": "",
    },
    "ND16": {
        "long_name": "Depth of the 16 deg C isotherm",
        "units": "m",
        "Notes": "",
    },
    "NDFR": {
        "long_name": "Depth of the lowest model level in the NCODA analysis",
        "units": "m",
        "Notes": "",
    },
    "NTFR": {
        "long_name": "Ocean T at the lowest level in the NCODA analysis",
        "units": "deg C*10",
        "Notes": "",
    },
    "NOHC": {
        "long_name": "Ocean heat content relative to the 26 C isotherm, from the NCODA analysis",
        "units": "J/kg-deg C",
        "Notes": "",
    },
    "NO20": {
        "long_name": "Ocean heat content relative to the 20 deg C isotherm, from the NCODA analysis",
        "units": "J/kg-deg C",
        "Notes": "",
    },
    "XNST": {
        "long_name": "Climatological values of the SST from the NCODA analysis",
        "units": "deg C*10",
        "Notes": "",
    },
    "XTMX": {
        "long_name": "Climatological values of the max ocean temperature in the NCODA vertical profile",
        "units": "deg C*10",
        "Notes": "",
    },
    "XDMX": {
        "long_name": "Climatological values of the depth of the max ocean temperature in the profile",
        "units": "m",
        "Notes": "",
    },
    "XDML": {
        "long_name": "Climatological values of the depth of the mixed layer",
        "units": "m",
        "Notes": """Defined as the depth where the T is 0.5 colder than at the surface. 
                    In rare cases, this is the depth where the T is 0.5 warmer than at the surface.
                    In those cases, XDML is negative.""",
    },
    "XD16": {
        "long_name": "Climatological values of the depth of the 16 deg C isotherm from the NCODA analysis",
        "units": "m",
        "Notes": "",
    },
    "XD18": {
        "long_name": "Climatological values of the depth of the 18 deg C isotherm from the NCODA analysis",
        "units": "m",
        "Notes": "",
    },
    "XD20": {
        "long_name": "Climatological values of the depth of the 20 deg C isotherm from the NCODA analysis",
        "units": "m",
        "Notes": "",
    },
    "XD22": {
        "long_name": "Climatological values of the depth of the 22 deg C isotherm from the NCODA analysis",
        "units": "m",
        "Notes": "",
    },
    "XD24": {
        "long_name": "Climatological values of the depth of the 24 deg C isotherm from the NCODA analysis",
        "units": "m",
        "Notes": "",
    },
    "XD26": {
        "long_name": "Climatological values of the depth of the 26 deg C isotherm from the NCODA analysis",
        "units": "m",
        "Notes": "",
    },
    "XD28": {
        "long_name": "Climatological values of the depth of the 28 deg C isotherm from the NCODA analysis",
        "units": "m",
        "Notes": "",
    },
    "XD30": {
        "long_name": "Climatological values of the depth of the 30 deg C isotherm from the NCODA analysis",
        "units": "m",
        "Notes": "",
    },
    "XDFR": {
        "long_name": "Climatological values of the depth of the lowest model level in the NCODA analysis",
        "units": "m",
        "Notes": "",
    },
    "XTFR": {
        "long_name": "Climatological values of the Ocean T at the lowest level in the NCODA analysis",
        "units": "deg C*10",
        "Notes": "",
    },
    "XOHC": {
        "long_name": "Climatological values of the Ocean heat content relative to the 26 C isotherm, from the NCODA analysis",
        "units": "KJ/cm2",
        "Notes": "",
    },
    "X020": {
        "long_name": "Climatological values of the Ocean heat content relative to the 20 C isotherm, from the NCODA analysis",
        "units": "J/kg-deg C",
        "Notes": "",
    },
    "XDST": {
        "long_name": "Climatological value of the daily Reynolds SST",
        "units": "deg C*10",
        "Notes": "",
    },
}

# %% Metpy units
[
    "%",
    "A",
    "A_90",
    "A_US",
    "A_it",
    "Ah",
    "At",
    "B",
    "BDFT",
    "BF",
    "BTU",
    "Ba",
    "Bd",
    "Bi",
    "Bq",
    "Btu",
    "Btu_iso",
    "Btu_it",
    "Btu_th",
    "C",
    "C_90",
    "Ci",
    "Cl",
    "Context",
    "D",
    "DPI",
    "Da",
    "ECC",
    "EC_therm",
    "E_h",
    "Eh",
    "F",
    "FBM",
    "F_90",
    "Fr",
    "G",
    "G_0",
    "Gal",
    "Gb",
    "Group",
    "Gy",
    "H",
    "H2O",
    "H_90",
    "Hg",
    "Hg_0C",
    "Hg_32F",
    "Hg_60F",
    "Hz",
    "J",
    "K",
    "KPH",
    "K_J",
    "K_J90",
    "K_alpha_Cu_d_220",
    "K_alpha_Mo_d_220",
    "K_alpha_W_d_220",
    "L",
    "Ly",
    "M",
    "MPH",
    "Measurement",
    "Mx",
    "N",
    "N_A",
    "Ne",
    "NeC",
    "Nm",
    "Np",
    "Oe",
    "P",
    "PPCM",
    "PPI",
    "PSH",
    "Pa",
    "Phi_0",
    "Quantity",
    "R",
    "RKM",
    "R_K",
    "R_K90",
    "R_inf",
    "R_∞",
    "Rd",
    "Ry",
    "S",
    "SPL",
    "St",
    "Sv",
    "System",
    "T",
    "Ta",
    "Td",
    "Tj",
    "Tt",
    "U",
    "UK_bbl",
    "UK_bushel",
    "UK_cup",
    "UK_cwt",
    "UK_fluid_ounce",
    "UK_force_ton",
    "UK_gallon",
    "UK_gill",
    "UK_horsepower",
    "UK_hundredweight",
    "UK_pint",
    "UK_pk",
    "UK_quart",
    "UK_ton",
    "UK_ton_force",
    "US_cwt",
    "US_dry_barrel",
    "US_dry_gallon",
    "US_dry_pint",
    "US_dry_quart",
    "US_fluid_dram",
    "US_fluid_ounce",
    "US_force_ton",
    "US_hundredweight",
    "US_international_ampere",
    "US_international_ohm",
    "US_international_volt",
    "US_liquid_cup",
    "US_liquid_dram",
    "US_liquid_fifth",
    "US_liquid_gallon",
    "US_liquid_gill",
    "US_liquid_ounce",
    "US_liquid_quart",
    "US_pint",
    "US_shot",
    "US_therm",
    "US_ton",
    "US_ton_force",
    "Unit",
    "UnitsContainer",
    "V",
    "VA",
    "V_90",
    "V_US",
    "V_it",
    "W",
    "W_90",
    "Wb",
    "Wh",
    "Xu_Cu",
    "Xu_Mo",
    "Z_0",
    "a",
    "a0",
    "a_0",
    "a_u_action",
    "a_u_current",
    "a_u_electric_field",
    "a_u_energy",
    "a_u_force",
    "a_u_intensity",
    "a_u_length",
    "a_u_mass",
    "a_u_temp",
    "a_u_time",
    "abA",
    "abC",
    "abF",
    "abH",
    "abS",
    "abV",
    "abampere",
    "abcoulomb",
    "aberdeen",
    "abfarad",
    "abhenry",
    "abmho",
    "abohm",
    "absiemens",
    "abvolt",
    "abΩ",
    "acre",
    "acre_feet",
    "acre_foot",
    "add_context",
    "alpha",
    "amp",
    "ampere",
    "ampere_hour",
    "ampere_turn",
    "amu",
    "angstrom",
    "angstrom_star",
    "angular_degree",
    "angular_minute",
    "angular_second",
    "ap_dr",
    "ap_lb",
    "ap_oz",
    "apothecary_dram",
    "apothecary_ounce",
    "apothecary_pound",
    "arc_minute",
    "arc_second",
    "arcdeg",
    "arcdegree",
    "arcmin",
    "arcminute",
    "arcsec",
    "arcsecond",
    "are",
    "astronomical_unit",
    "at",
    "atm",
    "atm_l",
    "atmosphere",
    "atmosphere_liter",
    "atomic_mass_constant",
    "atomic_unit_of_action",
    "atomic_unit_of_current",
    "atomic_unit_of_electric_field",
    "atomic_unit_of_energy",
    "atomic_unit_of_force",
    "atomic_unit_of_intensity",
    "atomic_unit_of_length",
    "atomic_unit_of_mass",
    "atomic_unit_of_temperature",
    "atomic_unit_of_time",
    "au",
    "auto_reduce_dimensions",
    "autoconvert_offset_to_baseunit",
    "autoconvert_to_preferred",
    "avdp_dram",
    "avdp_ounce",
    "avdp_pound",
    "avogadro_constant",
    "avogadro_number",
    "avoirdupois_dram",
    "avoirdupois_ounce",
    "avoirdupois_pound",
    "b",
    "bag",
    "bar",
    "barad",
    "barie",
    "barn",
    "barrel",
    "barrie",
    "baryd",
    "barye",
    "baud",
    "bbl",
    "becquerel",
    "beer_barrel",
    "beer_bbl",
    "big_point",
    "biot",
    "biot_turn",
    "bit",
    "bits_per_pixel",
    "blob",
    "board_feet",
    "board_foot",
    "bohr",
    "bohr_magneton",
    "bohr_radius",
    "boiler_horsepower",
    "boltzmann_constant",
    "bp",
    "bpp",
    "bps",
    "british_thermal_unit",
    "bu",
    "buckingham",
    "bushel",
    "byte",
    "c",
    "c_0",
    "c_1",
    "c_2",
    "cables_length",
    "cache_folder",
    "cal",
    "cal_15",
    "cal_it",
    "cal_th",
    "calorie",
    "candela",
    "candle",
    "carat",
    "case_sensitive",
    "cc",
    "cd",
    "celsius",
    "centimeter",
    "centimeter_H2O",
    "centimeter_Hg",
    "centimeter_Hg_0C",
    "centipoise",
    "centuries",
    "century",
    "chain",
    "characteristic_impedance_of_vacuum",
    "check",
    "cicero",
    "circle",
    "circular_mil",
    "classical_electron_radius",
    "clausius",
    "cmH2O",
    "cmHg",
    "cm_1",
    "cm_H2O",
    "cm_Hg",
    "cmil",
    "common_year",
    "conductance_quantum",
    "context",
    "conventional_ampere_90",
    "conventional_coulomb_90",
    "conventional_farad_90",
    "conventional_henry_90",
    "conventional_josephson_constant",
    "conventional_mercury",
    "conventional_ohm_90",
    "conventional_volt_90",
    "conventional_von_klitzing_constant",
    "conventional_water",
    "conventional_watt_90",
    "convert",
    "cooling_tower_ton",
    "coulomb",
    "coulomb_constant",
    "count",
    "counts_per_second",
    "cp",
    "cps",
    "css_pixel",
    "ct",
    "cu_ft",
    "cu_in",
    "cu_yd",
    "cubic_centimeter",
    "cubic_feet",
    "cubic_foot",
    "cubic_inch",
    "cubic_yard",
    "cup",
    "curie",
    "cwt",
    "cycle",
    "d",
    "dB",
    "dBm",
    "dBu",
    "d_220",
    "dalton",
    "darcy",
    "day",
    "debye",
    "decade",
    "decibel",
    "decibelmicrowatt",
    "decibelmilliwatt",
    "decimeter",
    "decitex",
    "default_as_delta",
    "default_format",
    "default_system",
    "define",
    "deg",
    "degC",
    "degF",
    "degK",
    "degR",
    "degRe",
    "degree",
    "degreeC",
    "degreeE",
    "degreeF",
    "degreeK",
    "degreeN",
    "degreeR",
    "degreeRe",
    "degree_Celsius",
    "degree_E",
    "degree_Fahrenheit",
    "degree_Kelvin",
    "degree_N",
    "degree_Rankine",
    "degree_Reaumur",
    "degree_Réaumur",
    "degree_east",
    "degree_north",
    "degreesE",
    "degreesN",
    "degrees_E",
    "degrees_N",
    "degrees_east",
    "degrees_north",
    "delta_celsius",
    "delta_degC",
    "delta_degF",
    "delta_degRe",
    "delta_degreeC",
    "delta_degreeF",
    "delta_degreeRe",
    "delta_degree_Celsius",
    "delta_degree_Fahrenheit",
    "delta_degree_Reaumur",
    "delta_degree_Réaumur",
    "delta_fahrenheit",
    "delta_reaumur",
    "delta_réaumur",
    "den",
    "denier",
    "dgal",
    "didot",
    "dirac_constant",
    "disable_contexts",
    "dot",
    "dots_per_inch",
    "dpi",
    "dqt",
    "dr",
    "drachm",
    "dram",
    "dry_barrel",
    "dry_gallon",
    "dry_pint",
    "dry_quart",
    "dtex",
    "dwt",
    "dyn",
    "dyne",
    "e",
    "eV",
    "electric_constant",
    "electrical_horsepower",
    "electron_g_factor",
    "electron_mass",
    "electron_volt",
    "elementary_charge",
    "enable_contexts",
    "entropy_unit",
    "enzyme_unit",
    "enzymeunit",
    "eon",
    "eps0",
    "eps_0",
    "epsilon_0",
    "erg",
    "esu",
    "eu",
    "eulers_number",
    "fahrenheit",
    "farad",
    "faraday",
    "faraday_constant",
    "fathom",
    "feet",
    "feet_H2O",
    "femtometer",
    "fermi",
    "fifteen_degree_calorie",
    "fifth",
    "fine_structure_constant",
    "first_radiation_constant",
    "fldr",
    "floz",
    "fluid_dram",
    "fluid_ounce",
    "fluidram",
    "fm",
    "fmt_locale",
    "foot",
    "foot_H2O",
    "foot_per_second",
    "foot_pound",
    "footpound",
    "force_gram",
    "force_kilogram",
    "force_long_ton",
    "force_metric_ton",
    "force_ndarray",
    "force_ndarray_like",
    "force_ounce",
    "force_pound",
    "force_short_ton",
    "force_t",
    "force_ton",
    "fortnight",
    "fps",
    "franklin",
    "ft",
    "ftH2O",
    "ft_lb",
    "fur",
    "furlong",
    "g",
    "g0",
    "g_0",
    "g_e",
    "g_n",
    "gal",
    "galileo",
    "gallon",
    "gamma",
    "gamma_mass",
    "gauss",
    "get_base_units",
    "get_compatible_units",
    "get_dimensionality",
    "get_group",
    "get_name",
    "get_root_units",
    "get_symbol",
    "get_system",
    "gf",
    "gi",
    "gilbert",
    "gill",
    "gon",
    "gpm",
    "gr",
    "grad",
    "grade",
    "grain",
    "gram",
    "gram_force",
    "gravitational_constant",
    "gravity",
    "gray",
    "gregorian_year",
    "h",
    "ha",
    "hand",
    "hartree",
    "hartree_energy",
    "hbar",
    "hectare",
    "hectopascal",
    "henry",
    "hertz",
    "hogshead",
    "horsepower",
    "hour",
    "hp",
    "hr",
    "hundredweight",
    "hydraulic_horsepower",
    "impedance_of_free_space",
    "imperial_barrel",
    "imperial_bbl",
    "imperial_bu",
    "imperial_bushel",
    "imperial_cp",
    "imperial_cup",
    "imperial_fldr",
    "imperial_floz",
    "imperial_fluid_drachm",
    "imperial_fluid_dram",
    "imperial_fluid_ounce",
    "imperial_fluid_scruple",
    "imperial_gal",
    "imperial_gallon",
    "imperial_gi",
    "imperial_gill",
    "imperial_minim",
    "imperial_peck",
    "imperial_pint",
    "imperial_pk",
    "imperial_pt",
    "imperial_qt",
    "imperial_quart",
    "in",
    "inHg",
    "in_Hg",
    "inch",
    "inch_H2O_39F",
    "inch_H2O_60F",
    "inch_Hg",
    "inch_Hg_32F",
    "inch_Hg_60F",
    "inches",
    "international_british_thermal_unit",
    "international_calorie",
    "international_feet",
    "international_foot",
    "international_inch",
    "international_inches",
    "international_knot",
    "international_mile",
    "international_steam_table_calorie",
    "international_yard",
    "is_compatible_with",
    "jig",
    "josephson_constant",
    "joule",
    "julian_year",
    "jute",
    "k",
    "k_B",
    "k_C",
    "karat",
    "kat",
    "katal",
    "kayser",
    "kelvin",
    "kgf",
    "kilogram",
    "kilogram_force",
    "kilojoule",
    "kilometer",
    "kilometer_per_hour",
    "kilometer_per_second",
    "kip",
    "kip_per_square_inch",
    "knot",
    "knot_international",
    "kph",
    "kps",
    "ksi",
    "kt",
    "l",
    "lambda",
    "lambert",
    "langley",
    "lattice_spacing_of_Si",
    "lb",
    "lbf",
    "lbt",
    "league",
    "leap_year",
    "li",
    "light_year",
    "lightyear",
    "link",
    "liquid_cup",
    "liquid_gallon",
    "liquid_gill",
    "liquid_pint",
    "liquid_quart",
    "liter",
    "litre",
    "lm",
    "ln10",
    "load_definitions",
    "long_hundredweight",
    "long_ton",
    "long_ton_force",
    "lumen",
    "lunar_month",
    "lux",
    "lx",
    "ly",
    "m",
    "m_e",
    "m_n",
    "m_p",
    "m_u",
    "magnetic_constant",
    "magnetic_flux_quantum",
    "mas",
    "maxwell",
    "mean_international_ampere",
    "mean_international_ohm",
    "mean_international_volt",
    "mercury",
    "mercury_60F",
    "meter",
    "meter_per_second",
    "metre",
    "metric_horsepower",
    "metric_ton",
    "metric_ton_force",
    "mho",
    "mi",
    "microgram",
    "microliter",
    "micrometer",
    "micromole",
    "micron",
    "mil",
    "mil_length",
    "mile",
    "mile_per_hour",
    "millennia",
    "millennium",
    "milliarcsecond",
    "millibar",
    "milligram",
    "millimeter",
    "millimeter_Hg",
    "millimeter_Hg_0C",
    "min",
    "minim",
    "minute",
    "mmHg",
    "mm_Hg",
    "mol",
    "molar",
    "molar_gas_constant",
    "mole",
    "molec",
    "molecule",
    "month",
    "mph",
    "mpl_formatter",
    "mps",
    "mu0",
    "mu_0",
    "mu_B",
    "mu_N",
    "nautical_mile",
    "neper",
    "neutron_mass",
    "newton",
    "newtonian_constant_of_gravitation",
    "nit",
    "nmi",
    "non_int_type",
    "nuclear_magneton",
    "number_english",
    "number_meter",
    "oct",
    "octave",
    "octet",
    "oersted",
    "ohm",
    "ohm_90",
    "ohm_US",
    "ohm_it",
    "oil_barrel",
    "oil_bbl",
    "ounce",
    "ounce_force",
    "oz",
    "ozf",
    "ozt",
    "params",
    "parse_expression",
    "parse_pattern",
    "parse_unit_name",
    "parse_units",
    "parse_units_as_container",
    "parsec",
    "particle",
    "pascal",
    "pc",
    "pdl",
    "peak_sun_hour",
    "peck",
    "pel",
    "pennyweight",
    "percent",
    "perch",
    "pi",
    "pi_theorem",
    "pica",
    "picture_element",
    "pint",
    "pixel",
    "pixels_per_centimeter",
    "pixels_per_inch",
    "pk",
    "planck_constant",
    "planck_current",
    "planck_length",
    "planck_mass",
    "planck_temperature",
    "planck_time",
    "point",
    "poise",
    "pole",
    "pond",
    "pound",
    "pound_force",
    "pound_force_per_square_inch",
    "poundal",
    "pp",
    "ppi",
    "ppm",
    "preprocessors",
    "printers_dpi",
    "printers_pica",
    "printers_point",
    "proton_mass",
    "psi",
    "pt",
    "px",
    "qt",
    "quad",
    "quadrillion_Btu",
    "quart",
    "quarter",
    "r_e",
    "rad",
    "radian",
    "rads",
    "rankine",
    "rd",
    "reaumur",
    "reciprocal_centimeter",
    "refrigeration_ton",
    "rem",
    "remove_context",
    "revolution",
    "revolutions_per_minute",
    "revolutions_per_second",
    "reyn",
    "rhe",
    "rod",
    "roentgen",
    "rpm",
    "rps",
    "rutherford",
    "rydberg",
    "rydberg_constant",
    "réaumur",
    "röntgen",
    "s",
    "scaled_point",
    "scruple",
    "sec",
    "second",
    "second_radiation_constant",
    "section",
    "separate_format_defaults",
    "set_fmt_locale",
    "setup_matplotlib",
    "sft",
    "shake",
    "short_hundredweight",
    "short_ton",
    "short_ton_force",
    "shot",
    "sidereal_day",
    "sidereal_month",
    "sidereal_year",
    "siemens",
    "sievert",
    "sigma",
    "sigma_e",
    "slinch",
    "slm",
    "slpm",
    "slug",
    "slugette",
    "smi",
    "sound_pressure_level",
    "speed_of_light",
    "sq_deg",
    "sq_ft",
    "sq_in",
    "sq_mi",
    "sq_perch",
    "sq_pole",
    "sq_rod",
    "sq_yd",
    "sqdeg",
    "square_degree",
    "square_feet",
    "square_foot",
    "square_inch",
    "square_inches",
    "square_league",
    "square_mile",
    "square_rod",
    "square_survey_mile",
    "square_yard",
    "sr",
    "standard_atmosphere",
    "standard_gravity",
    "standard_liter_per_minute",
    "statA",
    "statC",
    "statF",
    "statH",
    "statT",
    "statV",
    "statWb",
    "statampere",
    "statcoulomb",
    "statfarad",
    "stathenry",
    "statmho",
    "statohm",
    "stattesla",
    "statvolt",
    "statweber",
    "statΩ",
    "stefan_boltzmann_constant",
    "steradian",
    "stere",
    "stilb",
    "stokes",
    "stone",
    "super_feet",
    "super_foot",
    "superficial_feet",
    "superficial_foot",
    "survey_foot",
    "survey_link",
    "survey_mile",
    "sv",
    "svedberg",
    "sverdrup",
    "synodic_month",
    "sys",
    "t",
    "tTNT",
    "t_force",
    "tablespoon",
    "tansec",
    "tbsp",
    "teaspoon",
    "technical_atmosphere",
    "tesla",
    "tex",
    "tex_cicero",
    "tex_didot",
    "tex_pica",
    "tex_point",
    "tf",
    "th",
    "therm",
    "thermochemical_british_thermal_unit",
    "thermochemical_calorie",
    "thm",
    "thomson_cross_section",
    "thou",
    "tlb",
    "toe",
    "ton",
    "ton_TNT",
    "ton_force",
    "ton_of_refrigeration",
    "tonne",
    "tonne_of_oil_equivalent",
    "torr",
    "townsend",
    "toz",
    "tropical_month",
    "tropical_year",
    "troy_ounce",
    "troy_pound",
    "tsp",
    "turn",
    "u",
    "unified_atomic_mass_unit",
    "unit_pole",
    "us_statute_mile",
    "vacuum_permeability",
    "vacuum_permittivity",
    "volt",
    "volt_ampere",
    "von_klitzing_constant",
    "water",
    "water_39F",
    "water_4C",
    "water_60F",
    "watt",
    "watt_hour",
    "watthour",
    "weber",
    "week",
    "wien_frequency_displacement_law_constant",
    "wien_u",
    "wien_wavelength_displacement_law_constant",
    "wien_x",
    "with_context",
    "wraps",
    "x_unit_Cu",
    "x_unit_Mo",
    "yard",
    "yd",
    "year",
    "yr",
    "zeta",
    "°C",
    "°F",
    "°K",
    "°R",
    "°Re",
    "µ",
    "µ_0",
    "µ_B",
    "µ_N",
    "Å",
    "Å_star",
    "ångström",
    "ørsted",
    "ħ",
    "Δcelsius",
    "ΔdegC",
    "ΔdegF",
    "ΔdegRe",
    "ΔdegreeC",
    "ΔdegreeF",
    "ΔdegreeRe",
    "Δdegree_Réaumur",
    "Δfahrenheit",
    "Δreaumur",
    "Δréaumur",
    "Δ°C",
    "Δ°F",
    "Δ°Re",
    "Φ_0",
    "Ω",
    "Ω_90",
    "Ω_US",
    "Ω_it",
    "α",
    "γ",
    "ε_0",
    "ζ",
    "λ",
    "μ",
    "π",
    "σ",
    "σ_e",
    "ℎ",
    "Å",
]
