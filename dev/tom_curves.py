# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt

from utils import toolbox

# %%
filepaths = [
    "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/HPsearch_01-13-14h36.pkl",
    "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/HPsearch_01-14-14h29.pkl",
]

results = []
for fp in filepaths:
    with open(fp, "rb") as f:
        results.append(pickle.load(f))

curve_train = []
curve_val = []
for res in results:
    train = np.array(res["train"])
    valid = np.array(res["validation"])
    min_idx = np.argmin(valid, axis=1)

    for idx, val in enumerate(min_idx):
        curve_train.append(train[idx][val])
        curve_val.append(valid[idx][val])

dropouts = np.linspace(0, 1.0, 41)[1:-1]
# dropouts = np.hstack([dropouts[20:], dropouts[0:20]])

# %% Baseline
with open(
    "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/MLR_TorchMLR_losses_12-12-15h00_panguweather_probabilistic.pkl",
    "rb",
) as f:
    MLR = pickle.load(f)

baseline_idx = np.argmin(MLR["validation"])
baseline_val = MLR["validation"][baseline_idx]
baseline_train = MLR["train"][baseline_idx]

# %%
fig, ax = plt.subplots(dpi=150)
ax.plot(dropouts, curve_train, label="Training")
ax.plot(dropouts, curve_val, label="Validation")
ax.axhline(baseline_train, color="C0", linestyle="--", label="MLR Training")
ax.axhline(baseline_val, color="C1", linestyle="--", label="MLR Validation")
ax.set_xlabel("Dropout Rate", color="white")
ax.set_ylabel("CRPS @ min validation loss iteration", color="white")
ax.legend()
toolbox.plot_facecolors(fig=fig, axes=ax)

# %%
plt.figure(dpi=150)
plt.plot(dropouts, curve_train, label="Training")
plt.plot(dropouts, curve_val, label="Validation")
plt.xlabel("Dropout Rate")
plt.ylabel("CRPS @ min validation loss iteration")
plt.legend()
# %%
