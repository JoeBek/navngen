#!/usr/bin/env python3
"""
run_campus_oracle_traj.py

Build oracle subsample trajectories for campus or airport trials using
subsample_trajectory.py.  For each trial, K random subsamples of N keypoints
are tried per pair; the trial with the lowest GT error (translation or
rotation) is kept.

Usage:
    # campus
    python3 scripts/run_campus_oracle_traj.py -N 50 -K 1000 [--metric translation|rotation]
    python3 scripts/run_campus_oracle_traj.py -N 50 -K 1000 --metric rotation
    python3 scripts/run_campus_oracle_traj.py -N 50 -K 1000 --trials 12

    # airport
    python3 scripts/run_campus_oracle_traj.py --dataset airport -N 50 -K 1000
    python3 scripts/run_campus_oracle_traj.py --dataset airport -N 50 -K 1000 \\
        --trials MSNV_0000_0000_0_250804_1853_00156

Outputs:
    campus:  assets/trajectories/campus/trial_XX_oracle_{trans|rot}.tum
    airport: assets/trajectories/airport/{trial}_oracle_{trans|rot}.tum
"""

import sys
import argparse
import subprocess
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

CONFIG = project_root / "scripts/configs/camera_forward.yaml"

# campus
CAMPUS_FRAMES_DIR = project_root / "assets/outputs/campus_frames"
CAMPUS_GT_DIR     = project_root / "assets/trajectories/campus"
CAMPUS_OUT_DIR    = project_root / "assets/trajectories/campus"
CAMPUS_TRIALS     = [10, 11, 12]

# airport
AIRPORT_FRAMES_DIR = project_root / "assets/outputs/airport_frames"
AIRPORT_DATA_DIR   = Path("/home/joe/data/vt/airport")
AIRPORT_OUT_DIR    = project_root / "assets/trajectories/airport"
AIRPORT_TRIALS     = ["MSNV_0000_0000_0_250804_1853_00156"]

METRIC_SHORT = {"translation": "trans", "rotation": "rot"}


def resolve_paths(dataset, trial, short):
    """Return (frames_path, gt_path, out_path) for a single trial."""
    if dataset == "campus":
        t = int(trial)
        return (
            CAMPUS_FRAMES_DIR / f"campus_trial_{t:02d}_both_frames.pkl.gz",
            CAMPUS_GT_DIR     / f"trial_{t:02d}_gt.tum",
            CAMPUS_OUT_DIR    / f"trial_{t:02d}_oracle_{short}.tum",
        )
    else:  # airport
        return (
            AIRPORT_FRAMES_DIR / f"airport_{trial}_both_frames.pkl.gz",
            AIRPORT_DATA_DIR   / trial / "parquet" / "ground_truth_trajectory.tum",
            AIRPORT_OUT_DIR    / f"{trial}_oracle_{short}.tum",
        )


def build_parser():
    p = argparse.ArgumentParser(prog="run_campus_oracle_traj")
    p.add_argument("--dataset", default="campus", choices=["campus", "airport"])
    p.add_argument("-N", type=int, required=True, dest="N",
                   help="Keypoints to subsample per trial")
    p.add_argument("-K", type=int, default=1000, dest="K",
                   help="Trials per frame pair (default: 1000)")
    p.add_argument("--metric", default="translation",
                   choices=["translation", "rotation"],
                   help="Oracle selection metric (default: translation)")
    p.add_argument("--trials", type=str, nargs="+", default=None,
                   help="Trials to process. Campus: integers (default: 10 11 12). "
                        "Airport: trial name (default: MSNV_0000_0000_0_250804_1853_00156).")
    p.add_argument("--seed", type=int, default=None)
    return p


def main():
    args = build_parser().parse_args()
    short = METRIC_SHORT[args.metric]

    if args.trials is None:
        args.trials = [str(t) for t in CAMPUS_TRIALS] if args.dataset == "campus" else AIRPORT_TRIALS

    out_dir = CAMPUS_OUT_DIR if args.dataset == "campus" else AIRPORT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Oracle subsample trajectory  dataset={args.dataset}  metric={args.metric}  N={args.N}  K={args.K}")
    print(f"Trials: {args.trials}")

    for trial in args.trials:
        frames_path, gt_path, out_path = resolve_paths(args.dataset, trial, short)

        print(f"\n── {args.dataset}  {trial} ──────────────────────────────────────────")

        cmd = [
            sys.executable, str(project_root / "scripts/subsample_trajectory.py"),
            str(frames_path), str(out_path),
            "-N", str(args.N), "-K", str(args.K),
            "--config_path", str(CONFIG), "--camera", "surfnav",
            "--traj_format", "tum",
            "--gt", str(gt_path), "--gt_format", "tum",
            "--metric", args.metric,
        ]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]

        subprocess.run(cmd, check=True)
        print(f"  → {out_path}")

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
