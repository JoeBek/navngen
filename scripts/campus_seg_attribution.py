#!/usr/bin/env python3
"""
campus_seg_attribution.py

For each consecutive frame pair in a campus trial, compute the keypoint
subsample difference map (prob_normalize(top_T) - prob_normalize(all_K)) and
accumulate the mean signed diff per Cityscapes segmentation class.

Positive = class over-represented in good trials (good keypoints live here).
Negative = class under-represented in good trials (bad keypoints live here).

Usage:
    python3 scripts/campus_seg_attribution.py -N 50 -K 10000 --top 100 \\
        [--metric translation|rotation|combined] \\
        [--trials 10 11 12] [--start S] [--end E] \\
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
    load_gt_tum, tum_relative_pose, compute_error,
)

FRAMES_DIR  = project_root / "assets/outputs/campus_frames"
GT_DIR      = project_root / "assets/trajectories/campus"
CONFIG      = project_root / "scripts/configs/camera_forward.yaml"
TRIALS_DIR  = Path("/home/joe/data/vt/airport/campus/trials")

TRIALS = [10, 11, 12]

N_CLASSES = 19
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]


def seg_mask_path(frame_path: Path, trial: int) -> Path:
    stem = frame_path.stem
    return TRIALS_DIR / f"trial_{trial:02d}" / "frames" / "seg_masks" / f"{stem}.npy"


def build_parser():
    p = argparse.ArgumentParser(prog="campus_seg_attribution")
    p.add_argument("-N",       type=int, required=True,  dest="N",
                   help="Keypoints to subsample per trial")
    p.add_argument("-K",       type=int, required=True,  dest="K",
                   help="Number of random trials per frame pair")
    p.add_argument("--top",    type=int, required=True,  metavar="T",
                   help="Keep only the T lowest-error trials for the diff map")
    p.add_argument("--metric", default="translation",
                   choices=["translation", "rotation", "combined"])
    p.add_argument("--trials", type=int, nargs="+", default=TRIALS)
    p.add_argument("--start",  type=int, default=None)
    p.add_argument("--end",    type=int, default=None)
    p.add_argument("--save",   type=Path, default=None,
                   help="Save histogram figure to this path (default: display)")
    p.add_argument("--data_out", type=Path, default=None,
                   help="Save raw per-class arrays to a .npz file")
    p.add_argument("--seed",   type=int, default=None)
    return p


def process_trial(trial, args, diff_sum, pixel_count):
    """Accumulate diff_sum[c] and pixel_count[c] in-place for one trial."""
    frames_path = FRAMES_DIR / f"campus_trial_{trial:02d}_both_frames.pkl.gz"
    gt_path     = GT_DIR     / f"trial_{trial:02d}_gt.tum"

    print(f"\n── trial {trial} ──────────────────────────────────────────")
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

        # GT relative pose
        gi, gj = nearest_gt(i), nearest_gt(j)
        if gi is None or gj is None:
            n_skipped += 1
            continue
        R_gt, t_gt = tum_relative_pose(gt_data, gi, gj)
        if np.linalg.norm(t_gt) < 1e-9:
            n_skipped += 1
            continue

        # Seg mask for frame i
        mask_path = seg_mask_path(Path(f_prev.path), trial)
        if not mask_path.exists():
            n_skipped += 1
            continue
        seg = np.load(mask_path)   # (H, W) uint8, Cityscapes IDs

        # K trials — track which match indices each trial used
        n_total_kpt = np.zeros(n_matches, dtype=np.int32)
        trial_results = []   # (err, match_indices)
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

        # Expected fraction of appearances that land in a top trial
        expected_ratio = len(top_trials) / len(valid)

        h, w = seg.shape
        # Per-match: score = (top appearances / total appearances) - expected
        for m in range(n_matches):
            if n_total_kpt[m] == 0:
                continue
            score = n_top_kpt[m] / n_total_kpt[m] - expected_ratio
            x = int(np.clip(round(kpts0[matches[m, 0], 0]), 0, w - 1))
            y = int(np.clip(round(kpts0[matches[m, 0], 1]), 0, h - 1))
            c = int(seg[y, x])
            diff_sum[c]    += score
            pixel_count[c] += 1

    print(f"  {end - start + 1 - n_skipped} pairs processed, {n_skipped} skipped")


def main():
    args = build_parser().parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    diff_sum    = np.zeros(N_CLASSES, dtype=np.float64)
    pixel_count = np.zeros(N_CLASSES, dtype=np.int64)

    for trial in args.trials:
        process_trial(trial, args, diff_sum, pixel_count)

    # Mean diff per class (zero if class never observed)
    mean_diff = np.where(pixel_count > 0, diff_sum / pixel_count, 0.0)

    # ── report ────────────────────────────────────────────────────────────────
    print(f"\n{'Class':20s}  {'Mean diff':>10s}  {'Pixel-frames':>14s}")
    print("  " + "-" * 50)
    for c in np.argsort(mean_diff)[::-1]:
        if pixel_count[c] > 0:
            print(f"  {CLASS_NAMES[c]:18s}  {mean_diff[c]:+10.6f}  {pixel_count[c]:14d}")

    if args.data_out is not None:
        np.savez(
            args.data_out,
            mean_diff=mean_diff,
            diff_sum=diff_sum,
            pixel_count=pixel_count,
            class_names=np.array(CLASS_NAMES),
        )
        print(f"\nRaw data → {args.data_out}")

    # ── plot ──────────────────────────────────────────────────────────────────
    present = pixel_count > 0
    indices = np.where(present)[0]
    values  = mean_diff[indices]
    labels  = [CLASS_NAMES[c] for c in indices]
    colors  = ["#d73027" if v > 0 else "#4575b4" for v in values]

    sort_order = np.argsort(values)[::-1]
    values = values[sort_order]
    labels = [labels[k] for k in sort_order]
    colors = [colors[k] for k in sort_order]

    trials_str = "+".join(str(t) for t in args.trials)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    ax.bar(range(len(labels)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("mean attribution score  (top-T rate − expected rate)")
    ax.set_title(
        f"Segmentation class attribution\n"
        f"trials={trials_str}  N={args.N}  K={args.K}  top={args.top}  metric={args.metric}"
    )
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Figure → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
