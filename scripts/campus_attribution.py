#!/usr/bin/env python3
"""
campus_attribution.py

For each consecutive frame pair in a campus or airport trial, run K random
subsamples, keep the top-T trials by pose error, and accumulate two probability
distributions over the good-trial keypoints:

  1. Segmentation class  — what fraction of good-trial keypoint appearances
                           fall in each Cityscapes class.
  2. Relative depth      — what fraction fall in each depth bin (0 = nearest,
                           1 = furthest within the frame, min-max normalised
                           per frame).

Each keypoint is weighted by how many top-T trials it appeared in, mirroring
what the top-T heatmap represents.

Usage:
    # campus
    python3 scripts/campus_attribution.py -N 50 -K 10000 --top 100 \\
        [--metric translation|rotation|combined] \\
        [--trials 10 11 12] [--start S] [--end E] \\
        [--depth_bins 50] [--save PATH] [--data_out PATH] [--seed INT]

    # airport
    python3 scripts/campus_attribution.py --dataset airport -N 50 -K 10000 --top 100 \\
        [--trials MSNV_0000_0000_0_250804_1853_00156]
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from src.navngen.export_trajectory import load_frames
from src.navngen.trajectory import Solver
from plot_keypoint_heatmap import (
    load_gt_tum, tum_relative_pose, compute_error,
)

CONFIG = project_root / "scripts/configs/camera_forward.yaml"

# campus
CAMPUS_FRAMES_DIR = project_root / "assets/outputs/campus_frames"
CAMPUS_GT_DIR     = project_root / "assets/trajectories/campus"
CAMPUS_TRIALS_DIR = Path("/home/joe/data/vt/airport/campus/trials")
CAMPUS_TRIALS     = ["10", "11", "12"]

# airport
AIRPORT_FRAMES_DIR = project_root / "assets/outputs/airport_frames"
AIRPORT_DATA_DIR   = Path("/home/joe/data/vt/airport")
AIRPORT_TRIALS     = ["MSNV_0000_0000_0_250804_1853_00156"]

N_CLASSES = 19
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]


def resolve_paths(dataset, trial):
    """Return (frames_path, gt_path, seg_mask_dir, depth_mask_dir)."""
    if dataset == "campus":
        t = int(trial)
        base = CAMPUS_TRIALS_DIR / f"trial_{t:02d}" / "frames"
        return (
            CAMPUS_FRAMES_DIR / f"campus_trial_{t:02d}_both_frames.pkl.gz",
            CAMPUS_GT_DIR     / f"trial_{t:02d}_gt.tum",
            base / "seg_masks",
            base / "depth_masks",
        )
    else:  # airport
        base = AIRPORT_DATA_DIR / trial / "downsampled_output_10fps_undistorted" / "frames"
        return (
            AIRPORT_FRAMES_DIR / f"airport_{trial}_both_frames.pkl.gz",
            AIRPORT_DATA_DIR   / trial / "parquet" / "ground_truth_trajectory.tum",
            base / "seg_masks",
            base / "depth_masks",
        )


def build_parser():
    p = argparse.ArgumentParser(prog="campus_attribution")
    p.add_argument("--dataset",    default="campus", choices=["campus", "airport"])
    p.add_argument("-N",           type=int, required=True, dest="N")
    p.add_argument("-K",           type=int, required=True, dest="K")
    p.add_argument("--top",        type=int, required=True, metavar="T")
    p.add_argument("--metric",     default="translation",
                   choices=["translation", "rotation", "combined"])
    p.add_argument("--trials",     type=str, nargs="+", default=None,
                   help="Trials to process. Campus: integers (default: 10 11 12). "
                        "Airport: trial name (default: MSNV_0000_0000_0_250804_1853_00156).")
    p.add_argument("--start",      type=int, default=None)
    p.add_argument("--end",        type=int, default=None)
    p.add_argument("--depth_bins", type=int, default=50)
    p.add_argument("--save",       type=Path, default=None)
    p.add_argument("--data_out",   type=Path, default=None)
    p.add_argument("--seed",       type=int, default=None)
    return p


def process_trial(trial, args, seg_counts, depth_hist):
    frames_path, gt_path, seg_dir, depth_dir = resolve_paths(args.dataset, trial)

    print(f"\n── {args.dataset}  {trial} ────────────────────────────────")
    frames  = load_frames(frames_path)
    gt_data = load_gt_tum(gt_path)
    solver  = Solver(CONFIG, config_type="surfnav")
    gt_ts   = np.array([e[0] for e in gt_data])

    start = args.start if args.start is not None else 0
    end   = min(args.end if args.end is not None else len(frames) - 2,
                len(frames) - 2)

    def nearest_gt(frame_idx):
        ts = getattr(frames[frame_idx], "timestamp", None)
        if ts is None:
            return None
        return int(np.argmin(np.abs(gt_ts - float(ts))))

    n_skipped = 0
    for i in tqdm(range(start, end + 1), desc=f"trial {trial}"):
        j = i + 1
        f_prev, f_curr = frames[i], frames[j]

        if f_prev.kpts is None or f_curr.kpts is None or f_curr.matches is None:
            n_skipped += 1
            continue

        kpts0   = np.asarray(f_prev.kpts.numpy() if hasattr(f_prev.kpts, "numpy") else f_prev.kpts)
        kpts1   = np.asarray(f_curr.kpts.numpy() if hasattr(f_curr.kpts, "numpy") else f_curr.kpts)
        matches = np.asarray(f_curr.matches.numpy() if hasattr(f_curr.matches, "numpy") else f_curr.matches)

        n_matches = len(matches)
        n_sample  = min(args.N, n_matches)
        if n_sample < 5:
            n_skipped += 1
            continue

        gi, gj = nearest_gt(i), nearest_gt(j)
        if gi is None or gj is None:
            n_skipped += 1
            continue
        R_gt, t_gt = tum_relative_pose(gt_data, gi, gj)
        if np.linalg.norm(t_gt) < 1e-9:
            n_skipped += 1
            continue

        stem       = Path(f_prev.path).stem
        seg_path   = seg_dir   / f"{stem}.npy"
        depth_path = depth_dir / f"{stem}.npy"
        if not seg_path.exists() or not depth_path.exists():
            n_skipped += 1
            continue

        seg   = np.load(seg_path)    # (H, W) uint8
        depth = np.load(depth_path)  # (H, W) float32, raw metres

        # Normalise depth to [0, 1] per frame
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth)

        h, w = seg.shape

        # K trials — track per-match top-T appearance count
        n_total_kpt = np.zeros(n_matches, dtype=np.int32)
        trial_results = []
        for _ in range(args.K):
            idx = np.random.choice(n_matches, size=n_sample, replace=False)
            sub = matches[idx]
            err = float("nan")
            try:
                pose, _ = solver.solve_relative_pose(kpts0[sub[:, 0]], kpts1[sub[:, 1]])
                err = compute_error(pose.R, pose.t, R_gt, t_gt, args.metric)
            except Exception:
                pass
            n_total_kpt[idx] += 1
            trial_results.append((err, idx))

        valid = [(e, idx) for e, idx in trial_results if np.isfinite(e)]
        if not valid:
            n_skipped += 1
            continue
        valid.sort(key=lambda t: t[0])
        top_trials = valid[:args.top]

        n_top_kpt = np.zeros(n_matches, dtype=np.int32)
        for _, idx in top_trials:
            n_top_kpt[idx] += 1

        # Accumulate weighted by top-T appearance count
        for m in range(n_matches):
            weight = n_top_kpt[m]
            if weight == 0:
                continue
            x = int(np.clip(round(kpts0[matches[m, 0], 0]), 0, w - 1))
            y = int(np.clip(round(kpts0[matches[m, 0], 1]), 0, h - 1))

            seg_counts[int(seg[y, x])] += weight

            d_val = depth_norm[y, x]
            bin_idx = min(int(d_val * args.depth_bins), args.depth_bins - 1)
            depth_hist[bin_idx] += weight

    print(f"  {end - start + 1 - n_skipped} pairs processed, {n_skipped} skipped")


def main():
    args = build_parser().parse_args()
    if args.trials is None:
        args.trials = CAMPUS_TRIALS if args.dataset == "campus" else AIRPORT_TRIALS
    if args.seed is not None:
        np.random.seed(args.seed)

    seg_counts = np.zeros(N_CLASSES,       dtype=np.float64)
    depth_hist = np.zeros(args.depth_bins, dtype=np.float64)

    for trial in args.trials:
        process_trial(trial, args, seg_counts, depth_hist)

    total_seg   = seg_counts.sum()
    total_depth = depth_hist.sum()
    seg_prob    = seg_counts / total_seg   if total_seg   > 0 else seg_counts
    depth_prob  = depth_hist / total_depth if total_depth > 0 else depth_hist

    # ── report ────────────────────────────────────────────────────────────────
    print(f"\n{'Class':20s}  {'Probability':>12s}  {'Count':>12s}")
    print("  " + "-" * 48)
    for c in np.argsort(seg_prob)[::-1]:
        if seg_counts[c] > 0:
            print(f"  {CLASS_NAMES[c]:18s}  {seg_prob[c]:12.4f}  {int(seg_counts[c]):12d}")

    if args.data_out is not None:
        bin_centres = (np.arange(args.depth_bins) + 0.5) / args.depth_bins
        np.savez(
            args.data_out,
            seg_prob=seg_prob,
            seg_counts=seg_counts,
            depth_prob=depth_prob,
            depth_hist=depth_hist,
            depth_bin_centres=bin_centres,
            class_names=np.array(CLASS_NAMES),
        )
        print(f"\nRaw data → {args.data_out}")

    # ── plot ──────────────────────────────────────────────────────────────────
    trials_str = "+".join(str(t) for t in args.trials)
    title_base = f"trials={trials_str}  N={args.N}  K={args.K}  top={args.top}  metric={args.metric}"

    present    = seg_counts > 0
    seg_labels = [CLASS_NAMES[c] for c in range(N_CLASSES) if present[c]]
    seg_vals   = seg_prob[present]
    sort_order = np.argsort(seg_vals)[::-1]
    seg_labels = [seg_labels[k] for k in sort_order]
    seg_vals   = seg_vals[sort_order]

    bin_centres = (np.arange(args.depth_bins) + 0.5) / args.depth_bins

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Keypoint attribution — top-T heatmap  ({args.dataset})\n{title_base}", fontsize=11)

    ax1.bar(range(len(seg_labels)), seg_vals, color="steelblue",
            edgecolor="white", linewidth=0.5)
    ax1.set_xticks(range(len(seg_labels)))
    ax1.set_xticklabels(seg_labels, rotation=40, ha="right", fontsize=9)
    ax1.set_ylabel("probability")
    ax1.set_title("Segmentation class distribution")

    ax2.bar(bin_centres, depth_prob, width=1.0 / args.depth_bins,
            color="steelblue", edgecolor="none", align="center")
    ax2.set_xlabel("relative depth  (0 = near,  1 = far)")
    ax2.set_ylabel("probability")
    ax2.set_title("Depth distribution")
    ax2.set_xlim(0, 1)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Figure → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
