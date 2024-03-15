import os
import argparse
import xarray as xr
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    help="The input .dat file",
    default="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/data/SHIPS_dats/5day/lsdiag_na_1982_2019_sat_ts.dat",
)

args = parser.parse_args()

# Define the input and output paths
input_path = args.input

# Read the .dat file
with open(input_path, "r") as file:
    data_dict = {}
    for line in file:
        data = line.split()

        # Check to see if the header line has been reached
        if data[-1] == "HEAD":
            # read in the header data
            data_dict["shortname"] = data[0]
            data_dict["year"] = data[1][:2]
            data_dict["month"] = data[1][2:4]
            data_dict["day"] = data[1][4:6]
            data_dict["hour"] = data[2]
            data_dict["max_wind"] = data[3]
            data_dict["lat"] = data[4]
            data_dict["lon"] = data[5]
            data_dict["mslp"] = data[6]
            data_dict["ATCF_ID"] = data[7]

        if data[-1] == "TIME":
            # These are the lead times
            data_dict["time"] = np.array(data[0:-1]).astype("timedelta64[h]")

        if "RSST" in data:
            # Read the data into a dictionary
            data_dict["rsst"] = {
                "data": np.array(data[0:-2]).astype(float),
                "info_data": data[-1],
            }

        if "HIST" in data:
            print(data)

        if data[-1] == "LAST":
            # TODO: write the data to an xarray dataset
            # print(data_dict)
            input()

            # Reset the data dictionary
            data_dict = {}
