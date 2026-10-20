#!/usr/bin/env python3
"""
plot_auc.py

Plot AUC@X pose-error curves from subsample_pose_eval.py / subsample.py output.

Per pair, the pose error is max(rotation_err_deg, translation_err_deg).  The
curve shows the cumulative fraction of pairs with pose error ≤ t for t in [0, X];
AUC@X is that curve's mean value (= ∫CDF(t)dt / X).

Three series are drawn per file:
    best-of-K   pose error using the trial with lowest combined error per pair
    median-of-K typical-case subsample
    baseline    pose error from all matched keypoints

Usage:
    python3 scripts/plot_auc.py <file_or_dir> [--threshold 5] [--save-dir DIR]
    python3 scripts/plot_auc.py assets/outputs/subsample_errors/kitti/
    python3 scripts/plot_auc.py kitti_00_N500_errors.npz --threshold 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def per_pair_pose_error(rot, trans, combined):
    """
    rot, trans : [M, K]   per-trial rotation / translation errors (degrees)
    combined   : [M, K]   per-trial sum (used to pick the best trial)
    Returns (best, median) — each [M] in degrees, pose_error = max(rot, trans).
    """
    pose = np.fmax(rot, trans)                    # [M, K]
    best_k    = np.nanargmin(np.where(np.isnan(combined), np.inf, combined), axis=1)
    best      = pose[np.arange(pose.shape[0]), best_k]
    median    = np.nanmedian(pose, axis=1)
    return best, median


def auc_curve(errors, threshold, n_points=500):
    """
    Returns (ts, cdf, auc) for the pose-error CDF up to `threshold` degrees.
    AUC = mean(cdf) over the evaluation grid (= ∫CDF/threshold).
    """
    e = errors[np.isfinite(errors)]
    if len(e) == 0:
        return np.array([0, threshold]), np.zeros(2), 0.0
    ts  = np.linspace(0, threshold, n_points)
    cdf = (e[None, :] <= ts[:, None]).mean(axis=1)
    auc = float(np.trapezoid(cdf, ts) / threshold)
    return ts, cdf, auc


def plot_one(npz_path: Path, threshold: float, save_path: Path | None):
    d = np.load(npz_path)
    fig, ax = plt.subplots(figsize=(7, 5))

    # Subsample format: per-trial arrays + baseline
    if "trial_rot" in d.files and "trial_t" in d.files:
        rot      = d["trial_rot"]
        trans    = d["trial_t"]
        combined = d["trial_combined"] if "trial_combined" in d.files else d["trial_errors"]
        base_rot = d["base_rot"]
        base_t   = d["base_t"]
        N, K     = int(d["N"]), int(d["K"])
        M        = rot.shape[0]

        best, median = per_pair_pose_error(rot, trans, combined)
        baseline     = np.fmax(base_rot, base_t)

        for errors, label, color in [
            (best,     "best-of-K",   "tab:green"),
            (median,   "median-of-K", "tab:blue"),
            (baseline, "baseline",    "tab:red"),
        ]:
            ts, cdf, auc = auc_curve(errors, threshold)
            ax.plot(ts, cdf, color=color, linewidth=2,
                    label=f"{label}  AUC@{threshold:g}° = {auc:.4f}")
        suffix = f"(M={M} pairs, N={N}, K={K})"

    # Baseline-only format: rot_errors + trans_errors
    elif "rot_errors" in d.files and "trans_errors" in d.files:
        rot      = d["rot_errors"]
        trans    = d["trans_errors"]
        baseline = np.fmax(rot, trans)
        M        = len(baseline)
        ts, cdf, auc = auc_curve(baseline, threshold)
        ax.plot(ts, cdf, color="tab:red", linewidth=2,
                label=f"baseline  AUC@{threshold:g}° = {auc:.4f}")
        suffix = f"(M={M} pairs)"

    else:
        raise ValueError(f"{npz_path}: unrecognised format (keys: {d.files})")

    ax.set_xlim(0, threshold)
    ax.set_ylim(0, 1)
    ax.set_xlabel("pose error threshold (degrees)")
    ax.set_ylabel("fraction of pairs with error ≤ threshold")
    ax.set_title(f"{npz_path.stem}  —  AUC@{threshold:g}°  {suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path,
                   help="An .npz file or a directory containing *_errors.npz")
    p.add_argument("--threshold", type=float, default=5.0,
                   help="AUC threshold in degrees (default: 5)")
    p.add_argument("--save-dir", type=Path, default=None,
                   help="Save PNGs here instead of displaying. "
                        "If omitted and input is a directory, save next to each npz.")
    args = p.parse_args()

    if args.input.is_dir():
        files = sorted(args.input.glob("*_errors.npz"))
        if not files:
            print(f"No *_errors.npz in {args.input}")
            sys.exit(1)
        save_dir = args.save_dir if args.save_dir is not None else args.input
        save_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            print(f"── {f.name}")
            out = save_dir / f"{f.stem}_auc{int(args.threshold)}.png"
            plot_one(f, args.threshold, out)
    else:
        out = (args.save_dir / f"{args.input.stem}_auc{int(args.threshold)}.png"
               if args.save_dir is not None else None)
        if args.save_dir is not None:
            args.save_dir.mkdir(parents=True, exist_ok=True)
        plot_one(args.input, args.threshold, out)


if __name__ == "__main__":
    main()
