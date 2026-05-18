#!/usr/bin/env python3
"""
run_campus_oracle_traj.py

Build oracle subsample trajectories for campus trials using subsample_trajectory.py.
For each trial, K random subsamples of N keypoints are tried per pair; the trial
with the lowest GT error (translation or rotation) is kept.

Usage:
    python3 scripts/run_campus_oracle_traj.py -N 50 -K 1000 [--metric translation|rotation]
    python3 scripts/run_campus_oracle_traj.py -N 50 -K 1000 --metric rotation
    python3 scripts/run_campus_oracle_traj.py -N 50 -K 1000 --trials 12

Outputs are written to assets/trajectories/campus/ as:
    trial_XX_oracle_trans.tum  (metric=translation)
    trial_XX_oracle_rot.tum    (metric=rotation)
"""

import sys
import argparse
import subprocess
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

FRAMES_DIR = project_root / "assets/outputs/campus_frames"
GT_DIR     = project_root / "assets/trajectories/campus"
CONFIG     = project_root / "scripts/configs/camera_forward.yaml"
OUT_DIR    = project_root / "assets/trajectories/campus"

TRIALS       = [10, 11, 12]
METRIC_SHORT = {"translation": "trans", "rotation": "rot"}


def build_parser():
    p = argparse.ArgumentParser(prog="run_campus_oracle_traj")
    p.add_argument("-N", type=int, required=True, dest="N",
                   help="Keypoints to subsample per trial")
    p.add_argument("-K", type=int, default=1000, dest="K",
                   help="Trials per frame pair (default: 1000)")
    p.add_argument("--metric", default="translation",
                   choices=["translation", "rotation"],
                   help="Oracle selection metric (default: translation)")
    p.add_argument("--trials", type=int, nargs="+", default=TRIALS,
                   help=f"Trials to process (default: {TRIALS})")
    p.add_argument("--seed", type=int, default=None)
    return p


def main():
    args = build_parser().parse_args()
    short = METRIC_SHORT[args.metric]

    print(f"Oracle subsample trajectory  metric={args.metric}  N={args.N}  K={args.K}")
    print(f"Trials: {args.trials}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for trial in args.trials:
        frames_path = FRAMES_DIR / f"campus_trial_{trial:02d}_both_frames.pkl.gz"
        gt_path     = GT_DIR     / f"trial_{trial:02d}_gt.tum"
        out_path    = OUT_DIR    / f"trial_{trial:02d}_oracle_{short}.tum"

        print(f"\n── trial {trial} ──────────────────────────────────────────────")

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

    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
