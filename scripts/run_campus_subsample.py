#!/usr/bin/env python3
"""
Run subsample pose evaluation for 10 equally spaced frames across campus trials 10, 11, 12.

Usage:
    python3 scripts/run_campus_subsample.py -N <keypoints> [-K 1000]
"""

import sys
import argparse
import subprocess
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

FRAMES_DIR = project_root / "assets/outputs/campus_frames"
GT_DIR     = project_root / "assets/trajectories/campus"
CONFIG     = project_root / "scripts/configs/camera_forward.yaml"
OUT_DIR    = project_root / "assets/outputs/subsample_errors/campus"

TRIALS       = [10, 11, 12]
N_SAMPLE     = 10   # equally spaced frames per trial

FRAME_COUNTS = {10: 1260, 11: 1259, 12: 1259}


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-N", type=int, required=True, dest="N",
                   help="Keypoints to subsample per trial")
    p.add_argument("-K", type=int, default=1000, dest="K",
                   help="Trials per frame pair (default: 1000)")
    p.add_argument("--seed", type=int, default=None)
    return p


def main():
    args = build_parser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for trial in TRIALS:
        n_frames = FRAME_COUNTS[trial]
        indices  = np.linspace(0, n_frames - 2, N_SAMPLE, dtype=int)
        for i in indices:
            jobs.append((trial, int(i)))

    print(f"Running {len(jobs)} jobs  (N={args.N}, K={args.K})")

    for trial, i in jobs:
        frames_path = FRAMES_DIR / f"campus_trial_{trial:02d}_both_frames.pkl.gz"
        gt_path     = GT_DIR     / f"trial_{trial:02d}_gt.tum"
        out_frames  = OUT_DIR    / f"trial_{trial:02d}_frame_{i:04d}.pkl.gz"
        out_errors  = OUT_DIR    / f"trial_{trial:02d}_frame_{i:04d}_errors.npz"

        print(f"\n── trial {trial}  frame {i} ──────────────────────────────")
        cmd = [
            sys.executable, str(project_root / "scripts/subsample_pose_eval.py"),
            str(frames_path), str(gt_path), str(out_frames),
            "--start", str(i), "--end", str(i + 1),
            "-N", str(args.N), "-K", str(args.K), "-L", "1",
            "--config_path", str(CONFIG), "--camera", "surfnav",
            "--gt_format", "tum",
            "--errors_out", str(out_errors),
        ]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]

        subprocess.run(cmd, check=True)

    print(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
