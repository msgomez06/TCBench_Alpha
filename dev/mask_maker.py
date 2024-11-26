# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt


def create_mask(
    radius_units,
    fade_radius=150 * 4,  # Radius of maximum wind * 4
    fade_func="linear",
):
    # Convert radius from units to pixels
    radius_pixels = radius_units / 25
    fradius_pixels = fade_radius / 25

    # Initialize a 241x241 array with zeros
    mask = np.zeros((241, 241), dtype=int)

    # Calculate the center of the array
    center_x, center_y = 120, 120

    # Create a meshgrid of x and y values
    x, y = np.meshgrid(np.arange(241), np.arange(241))

    # Calculate the distance from the center of the array
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Create a mask for the radius
    mask[distance <= radius_pixels] = 1

    reverse_dists = fradius_pixels + radius_pixels - distance
    reverse_dists[reverse_dists < 0] = 0
    reverse_dists[distance <= radius_pixels] = 0
    reverse_dists = reverse_dists / reverse_dists.max()
    if fade_func == "linear":
        mask = mask + reverse_dists
    elif fade_func == "exponential":
        exp_mask = 1 - np.exp(1 - reverse_dists * 2)
        exp_mask[exp_mask < 0] = 0
        exp_mask = exp_mask / exp_mask.max()
        mask = mask + exp_mask
    elif fade_func == "log":
        log_mask = np.log(1 + reverse_dists)
        log_mask[log_mask < 0] = 0
        log_mask = log_mask / log_mask.max()
        mask = mask + log_mask
    elif fade_func == "poly":
        quad_mask = (reverse_dists * 0.5) ** 4
        quad_mask[quad_mask < 0] = 0
        quad_mask = quad_mask / quad_mask.max()
        mask = mask + quad_mask
    elif fade_func == "root":
        root_mask = (reverse_dists * 0.5) ** (1 / 4)
        root_mask[root_mask < 0] = 0
        root_mask = root_mask / root_mask.max()
        mask = mask + root_mask

    mask[mask < 0] = 0

    return mask


def log_func(x, a, b, c, d):
    return a * np.log(b * x + c) + d


results_dir = (
    "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/"
)

unique_leadtimes = [6, 12, 18, 24, 48, 72, 96, 120, 144, 168]
log_params = np.load(f"{results_dir}log_params_84ptile.npy")

# Map the log function to the unique leadtimes
log_values = log_func(np.array(unique_leadtimes), *log_params) * 100


mask_dict = {}
# Create a mask for each leadtime and plot it
for masktype in ["linear", "exponential", "log", "poly", "root"]:
    mask_dict[masktype] = {}
    for i, leadtime in enumerate(unique_leadtimes):
        mask = create_mask(log_values[i], fade_func=masktype)
        mask_dict[masktype][leadtime] = mask

# save the masks
with open(f"{results_dir}mask_dict.pkl", "wb") as f:
    pickle.dump(mask_dict, f)


# %%
