#!/usr/bin/env python3
"""
kitti_attribution.py

For each consecutive frame pair in one or more KITTI odometry sequences, run K
random subsamples, keep the top-T trials by pose error, and accumulate two
probability distributions over the good-trial keypoints:

  1. Segmentation class  — fraction of good-trial keypoint appearances per
                           Cityscapes class.
  2. Relative depth      — fraction per depth bin (0 = near, 1 = far,
                           min-max normalised per frame).

Each keypoint is weighted by how many top-T trials it appeared in.

Paths expected:
  frames    assets/outputs/kitti/both/kitti_{seq}_both_frames.pkl.gz
  GT        assets/trajectories/kitti/{seq}.txt          (KITTI 12-val format)
  calib     /home/joe/data/kitti/color/dataset/sequences/{seq}/calib.txt
  seg       /home/joe/data/kitti/seg/{seq}/masks/{stem}.npy
  depth     /home/joe/data/kitti/depth/{seq}/masks/{stem}.npy

Usage:
    python3 scripts/kitti_attribution.py -N 50 -K 10000 --top 100 \\
        [--metric translation|rotation|combined] \\
        [--sequences 00 01 02] [--start S] [--end E] \\
        [--depth_bins 50] \\
        [--save PATH] [--data_out PATH] [--seed INT]
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
    load_gt_kitti, kitti_relative_pose, compute_error,
)

FRAMES_DIR  = project_root / "assets/outputs/kitti/both"
GT_DIR      = project_root / "assets/trajectories/kitti"
KITTI_ROOT  = Path("/home/joe/data/kitti")

SEQUENCES = [f"{i:02d}" for i in range(11)]   # 00–10

N_CLASSES = 19
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]


def seq_from_path(frame_path: Path) -> str:
    """Extract sequence number from a KITTI frame path.
    e.g. .../sequences/00/image_2/000000.png  →  '00'
    """
    return frame_path.parts[-3]


def build_parser():
    p = argparse.ArgumentParser(prog="kitti_attribution")
    p.add_argument("-N",           type=int, required=True, dest="N")
    p.add_argument("-K",           type=int, required=True, dest="K")
    p.add_argument("--top",        type=int, required=True, metavar="T")
    p.add_argument("--metric",     default="translation",
                   choices=["translation", "rotation", "combined"])
    p.add_argument("--sequences",  nargs="+", default=SEQUENCES,
                   help=f"Sequences to process (default: {SEQUENCES})")
    p.add_argument("--start",      type=int, default=None)
    p.add_argument("--end",        type=int, default=None)
    p.add_argument("--depth_bins", type=int, default=50)
    p.add_argument("--save",       type=Path, default=None)
    p.add_argument("--data_out",   type=Path, default=None)
    p.add_argument("--seed",       type=int, default=None)
    return p


def process_sequence(seq, args, seg_counts, depth_hist):
    frames_path = FRAMES_DIR / f"kitti_{seq}_both_frames.pkl.gz"
    gt_path     = GT_DIR     / f"{seq}.txt"
    calib_path  = KITTI_ROOT / "color/dataset/sequences" / seq / "calib.txt"
    seg_dir     = KITTI_ROOT / "seg"   / seq / "masks"
    depth_dir   = KITTI_ROOT / "depth" / seq / "masks"

    for p in (frames_path, gt_path, calib_path):
        if not p.exists():
            print(f"  [SKIP] seq {seq}: missing {p}")
            return

    print(f"\n── sequence {seq} ─────────────────────────────────────────")
    frames  = load_frames(frames_path)
    gt_data = load_gt_kitti(gt_path)
    solver  = Solver(calib_path, config_type="kitti")

    start = args.start if args.start is not None else 0
    end   = min(args.end if args.end is not None else len(frames) - 2,
                len(frames) - 2)

    n_skipped = 0
    for i in tqdm(range(start, end + 1), desc=f"seq {seq}"):
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

        if i >= len(gt_data) or j >= len(gt_data):
            n_skipped += 1
            continue
        R_gt, t_gt = kitti_relative_pose(gt_data, i, j)
        if np.linalg.norm(t_gt) < 1e-9:
            n_skipped += 1
            continue

        stem      = Path(f_prev.path).stem
        seg_path  = seg_dir   / f"{stem}.npy"
        dep_path  = depth_dir / f"{stem}.npy"
        if not seg_path.exists() or not dep_path.exists():
            n_skipped += 1
            continue

        seg   = np.load(seg_path)
        depth = np.load(dep_path)

        d_min, d_max = depth.min(), depth.max()
        depth_norm = (depth - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth)

        h, w = seg.shape

        # K trials
        n_total_kpt   = np.zeros(n_matches, dtype=np.int32)
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

        for m in range(n_matches):
            weight = n_top_kpt[m]
            if weight == 0:
                continue
            x = int(np.clip(round(kpts0[matches[m, 0], 0]), 0, w - 1))
            y = int(np.clip(round(kpts0[matches[m, 0], 1]), 0, h - 1))

            seg_counts[int(seg[y, x])] += weight

            bin_idx = min(int(depth_norm[y, x] * args.depth_bins), args.depth_bins - 1)
            depth_hist[bin_idx] += weight

    print(f"  {end - start + 1 - n_skipped} pairs processed, {n_skipped} skipped")


def main():
    args = build_parser().parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    seg_counts = np.zeros(N_CLASSES,       dtype=np.float64)
    depth_hist = np.zeros(args.depth_bins, dtype=np.float64)

    for seq in args.sequences:
        process_sequence(seq, args, seg_counts, depth_hist)

    total_seg   = seg_counts.sum()
    total_depth = depth_hist.sum()
    seg_prob   = seg_counts / total_seg   if total_seg   > 0 else seg_counts
    depth_prob = depth_hist / total_depth if total_depth > 0 else depth_hist

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
    seqs_str   = "+".join(args.sequences)
    title_base = f"seqs={seqs_str}  N={args.N}  K={args.K}  top={args.top}  metric={args.metric}"

    present    = seg_counts > 0
    seg_labels = [CLASS_NAMES[c] for c in range(N_CLASSES) if present[c]]
    seg_vals   = seg_prob[present]
    sort_order = np.argsort(seg_vals)[::-1]
    seg_labels = [seg_labels[k] for k in sort_order]
    seg_vals   = seg_vals[sort_order]

    bin_centres = (np.arange(args.depth_bins) + 0.5) / args.depth_bins

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Keypoint attribution — top-T heatmap  (KITTI)\n{title_base}", fontsize=11)

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
