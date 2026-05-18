#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde

parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("--save",  type=Path, default=None, metavar="PATH",
                    help="Save figure to PATH instead of displaying it")
parser.add_argument("--bins",  type=int, default=60,
                    help="Number of histogram bins (default: 60)")
parser.add_argument("--xlim",  type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="Fix x-axis range, e.g. --xlim 0 2")
parser.add_argument("--bw",     type=float, default=None,
                    help="KDE bandwidth factor (default: Scott's rule). "
                         "Smaller = tighter, e.g. --bw 0.1")
parser.add_argument("--metric", default=None, choices=["translation", "rotation"],
                    help="Plot only one metric (eval_baseline format only)")
args = parser.parse_args()

d = np.load(args.input)

def _hist(ax, vals, title, xlabel):
    v = vals[np.isfinite(vals)]
    if args.xlim:
        v = v[(v >= args.xlim[0]) & (v <= args.xlim[1])]
    weights = np.ones(len(v)) / len(v)
    _, bin_edges, _ = ax.hist(v, bins=args.bins, alpha=0.3,
                              weights=weights, color="steelblue")
    if len(v) > 1:
        bin_width = bin_edges[1] - bin_edges[0]
        kde = gaussian_kde(v, bw_method=args.bw)
        x = np.linspace(v.min(), v.max(), 500)
        ax.plot(x, kde(x) * bin_width, color="steelblue", linewidth=2)
    ax.axvline(np.mean(v), color="red", linestyle="--",
               label=f"mean {np.mean(v):.4f}  median {np.median(v):.4f}")
    if args.xlim:
        ax.set_xlim(args.xlim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability")
    ax.legend(fontsize=8)

# eval_baseline format: trans_errors + rot_errors per pair
if "trans_errors" in d and "rot_errors" in d:
    trans = d["trans_errors"]
    rot   = d["rot_errors"]
    M     = len(trans)
    if args.metric == "translation":
        fig, ax1 = plt.subplots(figsize=(7, 4))
        fig.suptitle(f"Baseline error distribution  —  {M} pairs")
        _hist(ax1, trans, "Translation error", "unit L2  [0, 2]")
    elif args.metric == "rotation":
        fig, ax2 = plt.subplots(figsize=(7, 4))
        fig.suptitle(f"Baseline error distribution  —  {M} pairs")
        _hist(ax2, rot, "Rotation error", "geodesic degrees")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Baseline error distribution  —  {M} pairs")
        _hist(ax1, trans, "Translation error", "unit L2  [0, 2]")
        _hist(ax2, rot,   "Rotation error",    "geodesic degrees")

# subsample format: trial_errors [M, K] + base_errors [M]
else:
    trial = d["trial_combined"] if "trial_combined" in d else d["trial_errors"]
    base  = d["base_combined"]  if "base_combined"  in d else d["base_errors"]
    N, K  = int(d["N"]), int(d["K"])
    M     = trial.shape[0]

    flat       = trial.ravel()
    flat       = flat[np.isfinite(flat)]
    base_valid = base[np.isfinite(base)]
    base_mean  = np.mean(base_valid)

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle(f"Subsample error distribution  —  {M} pairs, N={N}, K={K}")
    if args.xlim:
        flat       = flat[      (flat       >= args.xlim[0]) & (flat       <= args.xlim[1])]
        base_valid = base_valid[(base_valid >= args.xlim[0]) & (base_valid <= args.xlim[1])]
    weights = np.ones(len(flat)) / len(flat)
    _, bin_edges, _ = ax.hist(flat, bins=args.bins, alpha=0.3, weights=weights,
                              color="steelblue", label=f"subsampled ({len(flat)} trials)")
    if len(flat) > 1:
        bin_width = bin_edges[1] - bin_edges[0]
        kde = gaussian_kde(flat)
        x = np.linspace(flat.min(), flat.max(), 500)
        ax.plot(x, kde(x) * bin_width, color="steelblue", linewidth=2)
    ax.axvline(base_mean, color="red", linestyle="--", label=f"baseline {base_mean:.4f}")
    if args.xlim:
        ax.set_xlim(args.xlim)
    ax.set_xlabel("error")
    ax.set_ylabel("probability")
    ax.legend()

plt.tight_layout()

if args.save:
    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.save}")
else:
    plt.show()
