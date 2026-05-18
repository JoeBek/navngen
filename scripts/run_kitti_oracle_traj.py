#!/usr/bin/env python3
"""
run_kitti_oracle_traj.py

Build oracle subsample trajectories for KITTI sequences using subsample_trajectory.py.
For each sequence, K random subsamples of N keypoints are tried per pair; the trial
with the lowest GT error (translation or rotation) is kept.

Usage:
    python3 scripts/run_kitti_oracle_traj.py -N 50 -K 1000
    python3 scripts/run_kitti_oracle_traj.py -N 50 -K 1000 --metric rotation
    python3 scripts/run_kitti_oracle_traj.py -N 50 -K 1000 --seqs 0 1 2

Outputs are written to assets/trajectories/kitti/ as:
    00_oracle_trans.txt  (metric=translation)
    00_oracle_rot.txt    (metric=rotation)
"""

import sys
import argparse
import subprocess
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

FRAMES_DIR   = project_root / "assets/outputs/kitti/both"
GT_DIR       = project_root / "assets/trajectories/kitti"
OUT_DIR      = project_root / "assets/trajectories/kitti"
KITTI_DATA   = Path("/home/joe/data/kitti")

SEQS         = list(range(11))   # 00-10 have public GT
METRIC_SHORT = {"translation": "trans", "rotation": "rot"}


def build_parser():
    p = argparse.ArgumentParser(prog="run_kitti_oracle_traj")
    p.add_argument("-N", type=int, required=True, dest="N",
                   help="Keypoints to subsample per pair")
    p.add_argument("-K", type=int, default=1000, dest="K",
                   help="Trials per frame pair (default: 1000)")
    p.add_argument("--metric", default="translation",
                   choices=["translation", "rotation"],
                   help="Oracle selection metric (default: translation)")
    p.add_argument("--seqs", type=int, nargs="+", default=SEQS,
                   help=f"Sequences to process (default: {SEQS})")
    p.add_argument("--kitti_data", type=Path, default=KITTI_DATA,
                   help=f"Root KITTI data directory (default: {KITTI_DATA})")
    p.add_argument("--seed", type=int, default=None)
    return p


def main():
    args = build_parser().parse_args()
    short = METRIC_SHORT[args.metric]

    print(f"Oracle subsample trajectory  metric={args.metric}  N={args.N}  K={args.K}")
    print(f"Sequences: {args.seqs}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for seq in args.seqs:
        seq_str      = f"{seq:02d}"
        frames_path  = FRAMES_DIR / f"kitti_{seq_str}_both_frames.pkl.gz"
        gt_path      = GT_DIR     / f"{seq_str}.txt"
        calib_path   = args.kitti_data / "color" / "dataset" / "sequences" / seq_str / "calib.txt"
        out_path     = OUT_DIR    / f"{seq_str}_oracle_{short}.txt"

        print(f"\n── seq {seq_str} ──────────────────────────────────────────────")

        for label, path in [("frames", frames_path), ("GT", gt_path), ("calib", calib_path)]:
            if not path.exists():
                print(f"  [SKIP] {label} not found: {path}")
                break
        else:
            cmd = [
                sys.executable, str(project_root / "scripts/subsample_trajectory.py"),
                str(frames_path), str(out_path),
                "-N", str(args.N), "-K", str(args.K),
                "--config_path", str(calib_path), "--camera", "kitti",
                "--traj_format", "kitti",
                "--gt", str(gt_path), "--gt_format", "kitti",
                "--metric", args.metric,
            ]
            if args.seed is not None:
                cmd += ["--seed", str(args.seed)]

            subprocess.run(cmd, check=True)
            print(f"  → {out_path}")

    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
