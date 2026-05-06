#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("errors.npz")
d = np.load(path)

trial = d["trial_combined"]   # [M, K]
base  = d["base_combined"]    # [M]
N, K  = int(d["N"]), int(d["K"])
M     = trial.shape[0]

flat       = trial.ravel()
flat       = flat[np.isfinite(flat)]
base_valid = base[np.isfinite(base)]
base_mean  = np.mean(base_valid)

fig, ax = plt.subplots(figsize=(7, 4))
fig.suptitle(f"Subsample error distribution  —  {M} pairs, N={N}, K={K}")

ax.hist(flat, bins=60, alpha=0.7, label=f"subsampled ({len(flat)} trials)")
ax.axvline(base_mean, color="red", linestyle="--", label=f"baseline {base_mean:.1f}°")
ax.set_xlabel("combined error (°)")
ax.set_ylabel("count")
ax.legend()

plt.tight_layout()
plt.show()
