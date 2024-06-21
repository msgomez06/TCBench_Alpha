import os
import argparse
import xarray as xr
import numpy as np

# TCBench Libraries
from . import constants

# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    help="The input .dat file",
    default="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/data/SHIPS_dats/5day/lsdiag_na_1982_2019_sat_ts.dat",
)
parser.add_argument(
    "--output_folder",
    help="The output folder path",
    default="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/data/SHIPS_netcdfs/",
)

args = parser.parse_args()

# Define the input and output paths
input_path = args.input
output_folder = args.output_folder
# %%
# Read the .dat file
with open(input_path, "r") as file:
    data_dict = {}
    ds = None
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
            data_dict["datavars"] = []

            # If the dataset is None, create a new dataset
            if ds is None:
                ds = xr.Dataset()
                ds.attrs["ATCF_ID"] = data_dict["ATCF_ID"]
                print(ds.attrs["ATCF_ID"], flush=True)
            # if the dataset exists, check if the ATCF_ID is the same
            elif ds.attrs["ATCF_ID"] != data_dict["ATCF_ID"]:

                # If the ATCF_ID is different, check if the file
                # already exists. If it does, print that it does.
                # If it does not, write the dataset to a file
                if os.path.exists(
                    os.path.join(output_folder, f"{ds.attrs['ATCF_ID']}.nc")
                ):
                    print(
                        f"File {ds.attrs['ATCF_ID']}.nc already exists. Skipping.",
                        flush=True,
                    )
                else:
                    # write the dataset to a file
                    ds.to_netcdf(
                        os.path.join(output_folder, f"{ds.attrs['ATCF_ID']}.nc"),
                        mode="w",
                    )
                ds = xr.Dataset()
                ds.attrs["ATCF_ID"] = data_dict["ATCF_ID"]
                print(ds.attrs["ATCF_ID"], flush=True)

        if data[-1] == "TIME":
            # These are the lead times
            data_dict["lead_time"] = np.array(data[0:-1]).astype(np.dtype("i1"))
            # When the lead times are used in the code, it will be necessary
            # to change the lead times to timedelta[h] format

        # check if the line corresponds to a known variable type
        if np.any(np.isin(data, list(constants.SHIPS_metadata.keys()))):
            varname_idx = np.where(
                np.isin(data, list(constants.SHIPS_metadata.keys()))
            )[0].item()
            varname = data[varname_idx]
            vardata = np.array(data[:varname_idx]).astype(float)
            vardata[vardata == 9999] = np.nan
            attrs = constants.SHIPS_metadata[varname]
            data_length = int(attrs.get("Data_Length", len(vardata)))
            vardata = vardata[:data_length]
            if "Info_Label" in list(attrs.keys()):
                attrs["Info_Data"] = data[-1]

            data_dict["datavars"].append({varname: {"data": vardata, "attrs": attrs}})

        # if "RSST" in data:
        #     # Read the data into a dictionary
        #     data_dict["rsst"] = {
        #         "data": np.array(data[0:-2]).astype(float),
        #         "info_data": data[-1],
        #     }

        # if "HIST" in data:
        #     print(data)

        if data[-1] == "LAST":
            # TODO: write the data to the xarray dataset
            temp_ds = xr.Dataset()
            if int(data_dict["year"]) >= 82:
                year = f'19{data_dict["year"]}'
            else:
                year = f'20{data_dict["year"]}'
            timestamp = np.datetime64(
                f'{year}-{data_dict["month"]}-{data_dict["day"]}T{data_dict["hour"]}'
            ).astype("datetime64[ns]")

            # Iterate over the data variables
            for var in data_dict["datavars"]:
                for key in var.keys():
                    if var[key]["attrs"].get("LeadTimeVar", True):
                        coords = {
                            "t": [timestamp],
                            "lead_time": data_dict["lead_time"][
                                -var[key]["data"].size :
                            ],  # Only take the lead times up to the length of the data
                        }
                    else:
                        coords = {
                            "t": [timestamp],
                            "levels": np.arange(1, len(var[key]["data"]) + 1),
                        }

                    if "LeadTimeVar" in list(var[key]["attrs"].keys()):
                        var[key]["attrs"].pop("LeadTimeVar")

                    temp_ds = temp_ds.assign(
                        {
                            key: xr.DataArray(
                                var[key]["data"][
                                    None, :
                                ],  # Add a new dimension to the data
                                # dims=coords.keys(),
                                coords=coords,
                                attrs=var[key]["attrs"],
                            )
                        }
                    )
                # Add the lat, lon, max wind and mslp as single level vars
                temp_ds = temp_ds.assign(
                    {
                        "lat": xr.DataArray(
                            [data_dict["lat"]],
                            coords={"t": [timestamp]},
                            attrs={"long_name": "Latitude", "units": "degrees_north"},
                        ),
                        "lon": xr.DataArray(
                            [data_dict["lon"]],
                            coords={"t": [timestamp]},
                            attrs={"long_name": "Longitude", "units": "degrees_west"},
                        ),
                        "max_wind": xr.DataArray(
                            [data_dict["max_wind"]],
                            coords={"t": [timestamp]},
                            attrs={"long_name": "Maximum Wind Speed", "units": "knots"},
                        ),
                        "mslp": xr.DataArray(
                            [data_dict["mslp"]],
                            coords={"t": [timestamp]},
                            attrs={
                                "long_name": "Mean Sea Level Pressure",
                                "units": "hPa",
                            },
                        ),
                    }
                )

            ds = xr.merge([ds, temp_ds])

            # Reset the data dictionary
            data_dict = {}

# %%
