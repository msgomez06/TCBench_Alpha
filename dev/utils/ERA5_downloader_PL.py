#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 11:44:20 2022

Python script used to download ERA5 data from the CDS API into single files
with multiple pressure levels.

@author: mgomezd1
"""

import cdsapi
import os
import numpy as np

folder_path = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ECMWF/ERA5/"

# load client interface
client = cdsapi.Client(timeout=600, quiet=False, debug=True)


# Single Pressure
data_origin = "reanalysis-era5-pressure-levels"

datavars = [
    # "divergence",
    # "fraction_of_cloud_cover",
    # "geopotential",
    # "ozone_mass_mixing_ratio",
    # "potential_vorticity",
    # "relative_humidity",
    # "specific_cloud_ice_water_content",
    # "specific_cloud_liquid_water_content",
    # "specific_humidity",
    # "specific_rain_water_content",
    # "specific_snow_water_content",
    # "temperature",
    # "u_component_of_wind",
    # "v_component_of_wind",
    # "vertical_velocity",
    "vorticity",
]

# Specify the pressure levels to download for each variable. Else, download all
plevels = {
    "vorticity": [
        "10",
        "20",
        "30",
        "50",
        "70",
        "100",
        "150",
        "200",
        "250",
        "300",
        "400",
        "500",
        "600",
        "700",
        "800",
        "850",
        "900",
        "925",
        "950",
        "975",
        "1000",
    ],
}

full_pressure = [
    "1",
    "2",
    "3",
    "5",
    "7",
    "10",
    "20",
    "30",
    "50",
    "70",
    "100",
    "125",
    "150",
    "175",
    "200",
    "225",
    "250",
    "300",
    "350",
    "400",
    "450",
    "500",
    "550",
    "600",
    "650",
    "700",
    "750",
    "775",
    "800",
    "825",
    "850",
    "875",
    "900",
    "925",
    "950",
    "975",
    "1000",
]

times = [
    "00:00",
    "03:00",
    "06:00",
    "09:00",
    "12:00",
    "15:00",
    "18:00",
    "21:00",
]

years = {
    "vorticity": [
        2005,
    ],
}

for var in datavars:
    for year in years.get(var, np.arange(1995, 2001, 1)):
        data_params = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": var,
            "pressure_level": plevels.get(var, full_pressure),
            "year": f"{year}",
            "month": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
            ],
            "day": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "30",
                "31",
            ],
            "time": times,
        }

        # Make sure the target folder exists
        directory_path = os.path.join(folder_path, "PL", var)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        # filename
        filename = f"ERA5_{year}_{var}.nc"
        target_path = os.path.join(directory_path, filename)
        client.retrieve(name=data_origin, request=data_params, target=target_path)
