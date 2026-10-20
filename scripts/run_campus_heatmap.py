#!/usr/bin/env python3
"""
run_campus_heatmap.py

Generate keypoint subsample heatmaps for every consecutive frame pair in one or
more campus or airport trials.  Saves one PNG per pair under:

    assets/outputs/heatmaps/{dataset}/{trial}/[NAME/]frame_XXXX.png

Frames are loaded once per trial so the pkl.gz is not reloaded on every frame.

Usage:
    # campus
    python3 scripts/run_campus_heatmap.py -N 50 -K 10000 --top 100 \\
        [--metric translation|rotation|combined] \\
        [--cmap NAME] [--trials 10 11 12] [--start S] [--end E] [--seed INT]

    # airport
    python3 scripts/run_campus_heatmap.py --dataset airport -N 50 -K 10000 --top 100
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
from src.navngen.load_images import load_image
from src.navngen.trajectory import Solver
from plot_keypoint_heatmap import (
    build_heatmap, image_tensor_to_rgb, overlay_heatmap, overlay_diff,
    diff_maps, load_gt_tum, tum_relative_pose, compute_error,
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


def resolve_paths(dataset, trial):
    """Return (frames_path, gt_path) for the given dataset and trial."""
    if dataset == "campus":
        t = int(trial)
        return (
            CAMPUS_FRAMES_DIR / f"campus_trial_{t:02d}_both_frames.pkl.gz",
            CAMPUS_GT_DIR     / f"trial_{t:02d}_gt.tum",
        )
    else:  # airport
        return (
            AIRPORT_FRAMES_DIR / f"airport_{trial}_both_frames.pkl.gz",
            AIRPORT_DATA_DIR   / trial / "parquet" / "ground_truth_trajectory.tum",
        )


def build_parser():
    p = argparse.ArgumentParser(prog="run_campus_heatmap")
    p.add_argument("--dataset", default="campus", choices=["campus", "airport"])
    p.add_argument("-N",       type=int, required=True,  dest="N",
                   help="Keypoints to subsample per trial")
    p.add_argument("-K",       type=int, required=True,  dest="K",
                   help="Number of random trials per frame pair")
    p.add_argument("--top",    type=int, default=None,   metavar="T",
                   help="Keep only the T lowest-error trials for the heatmap")
    p.add_argument("--metric", default="translation",
                   choices=["translation", "rotation", "combined"],
                   help="Error metric for --top ranking (default: translation)")
    p.add_argument("--cmap",   default="jet",
                   help="Matplotlib colormap (default: jet)")
    p.add_argument("--sigma",  type=float, default=15.0,
                   help="Gaussian smoothing sigma in pixels (default: 15)")
    p.add_argument("--alpha",  type=float, default=0.55,
                   help="Max heatmap opacity 0-1 (default: 0.55)")
    p.add_argument("--trials", type=str, nargs="+", default=None,
                   help="Trials to process. Campus: integers (default: 10 11 12). "
                        "Airport: trial name (default: MSNV_0000_0000_0_250804_1853_00156).")
    p.add_argument("--start",  type=int, default=None,
                   help="First frame index (default: 0)")
    p.add_argument("--end",    type=int, default=None,
                   help="Last frame index inclusive (default: last pair)")
    p.add_argument("--diff",   action="store_true",
                   help="Also save a difference map (top_T − all_K). Requires --top.")
    p.add_argument("--ransac", action="store_true",
                   help="Only include RANSAC inlier keypoints in the heatmap "
                        "(always runs the solver, even without --top).")
    p.add_argument("--name",   type=str, default=None,
                   help="Subdirectory inside trial_XX/ to group this run, e.g. --name top100_jet")
    p.add_argument("--seed",   type=int, default=None)
    return p


def process_trial(trial, args):
    frames_path, gt_path = resolve_paths(args.dataset, trial)
    out_dir = project_root / "assets/outputs/heatmaps" / args.dataset / str(trial)
    if args.name:
        out_dir = out_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n── {args.dataset}  {trial}  →  {out_dir} ────────────────")
    frames  = load_frames(frames_path)
    gt_data = load_gt_tum(gt_path)
    solver  = Solver(CONFIG, config_type="surfnav")

    gt_ts = np.array([e[0] for e in gt_data])

    start = args.start if args.start is not None else 0
    end   = min(args.end if args.end is not None else len(frames) - 2,
                len(frames) - 2)

    def nearest_gt(frame_idx):
        ts = getattr(frames[frame_idx], "timestamp", None)
        if ts is None:
            return None
        return int(np.argmin(np.abs(gt_ts - float(ts))))

    n_saved = 0
    for i in tqdm(range(start, end + 1), desc=f"trial {trial}"):
        j = i + 1
        f_prev, f_curr = frames[i], frames[j]

        if f_prev.kpts is None or f_curr.kpts is None or f_curr.matches is None:
            continue

        kpts0   = np.asarray(f_prev.kpts.numpy() if hasattr(f_prev.kpts, "numpy") else f_prev.kpts)
        kpts1   = np.asarray(f_curr.kpts.numpy() if hasattr(f_curr.kpts, "numpy") else f_curr.kpts)
        matches = np.asarray(f_curr.matches.numpy() if hasattr(f_curr.matches, "numpy") else f_curr.matches)

        n_matches = len(matches)
        n_sample  = min(args.N, n_matches)
        if n_sample < 5:
            continue

        # GT relative pose for --top filtering
        R_gt = t_gt = None
        if args.top is not None:
            gi, gj = nearest_gt(i), nearest_gt(j)
            if gi is not None and gj is not None:
                R_gt, t_gt = tum_relative_pose(gt_data, gi, gj)
            if R_gt is None or np.linalg.norm(t_gt) < 1e-9:
                continue

        # K trials — always keep all_trials for diff baseline
        all_trials = []
        need_solver = args.top is not None or args.diff or args.ransac
        for _ in range(args.K):
            idx = np.random.choice(n_matches, size=n_sample, replace=False)
            sub = matches[idx]
            c0  = kpts0[sub[:, 0]]
            c1  = kpts1[sub[:, 1]]
            err = None
            if need_solver:
                try:
                    pose, info = solver.solve_relative_pose(c0, c1)
                    err = compute_error(pose.R, pose.t, R_gt, t_gt, args.metric)
                    if args.ransac:
                        inliers = np.asarray(info["inliers"], dtype=bool)
                        c0, c1  = c0[inliers], c1[inliers]
                except Exception:
                    err = float("nan")
                    if args.ransac:
                        c0, c1 = np.empty((0, 2)), np.empty((0, 2))
            if not args.ransac or len(c0) > 0:
                all_trials.append((err, c0, c1))

        display_trials = all_trials
        if args.top is not None:
            valid = [(e, c0, c1) for e, c0, c1 in all_trials if np.isfinite(e)]
            if not valid:
                continue
            valid.sort(key=lambda t: t[0])
            display_trials = valid[:args.top]

        coords = [c0 for _, c0, _ in display_trials if len(c0) > 0]
        if not coords:
            continue
        selected0  = np.concatenate(coords, axis=0)
        img0       = image_tensor_to_rgb(load_image(f_prev.path))
        h0, w0     = img0.shape[:2]
        hmap_top   = build_heatmap(selected0, h0, w0, args.sigma)

        n_used     = len(display_trials)
        filter_str = (f"top {n_used} of {args.K} by {args.metric}"
                      if args.top is not None else f"{args.K} trials")

        if args.diff:
            all_coords0 = np.concatenate([c0 for _, c0, _ in all_trials if len(c0) > 0], axis=0)
            hmap_all    = build_heatmap(all_coords0, h0, w0, args.sigma)
            diff        = diff_maps(hmap_top, hmap_all)

            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(
                f"Keypoint heatmap & difference  —  N={args.N}, {filter_str}\n"
                f"frame {i}  |  {n_matches} total matches",
                fontsize=11,
            )
            overlay_heatmap(ax0, img0, hmap_top, args.alpha, args.cmap,
                            f"Top-{n_used} heatmap")
            sm0 = plt.cm.ScalarMappable(cmap=args.cmap, norm=plt.Normalize(vmin=0, vmax=1))
            sm0.set_array([])
            fig.colorbar(sm0, ax=ax0, fraction=0.025, pad=0.02,
                         label="relative selection frequency")

            overlay_diff(ax1, img0, diff, args.alpha,
                         "Difference  (top−all)  —  red=over, blue=under")
            sm1 = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-1, vmax=1))
            sm1.set_array([])
            fig.colorbar(sm1, ax=ax1, fraction=0.025, pad=0.02,
                         label="top − all  (normalised)")
        else:
            fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
            fig.suptitle(
                f"Keypoint heatmap  —  N={args.N}, {filter_str}\n"
                f"frame {i}  |  {n_matches} total matches",
                fontsize=11,
            )
            overlay_heatmap(ax0, img0, hmap_top, args.alpha, args.cmap, f"Frame {i}")
            sm = plt.cm.ScalarMappable(cmap=args.cmap, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            fig.colorbar(sm, ax=ax0, fraction=0.025, pad=0.02,
                         label="relative selection frequency")

        plt.tight_layout()
        out_path = out_dir / f"frame_{i:04d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        n_saved += 1

    print(f"  {n_saved} heatmaps saved → {out_dir}")


def main():
    args = build_parser().parse_args()
    if args.diff and args.top is None:
        print("Error: --diff requires --top")
        sys.exit(1)
    if args.trials is None:
        args.trials = CAMPUS_TRIALS if args.dataset == "campus" else AIRPORT_TRIALS
    if args.seed is not None:
        np.random.seed(args.seed)

    for trial in args.trials:
        process_trial(trial, args)

    out_base = project_root / "assets/outputs/heatmaps" / args.dataset
    print(f"\nDone. Outputs in {out_base}")


if __name__ == "__main__":
    main()
