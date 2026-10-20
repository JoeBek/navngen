#!/usr/bin/env python3
"""
run_kitti_subsample.py

Run subsample pose evaluation over all frame pairs in each KITTI sequence.
Produces per-trial outputs:
  kitti_{seq}_N{N}_errors.npz        — full error distributions for plotting
  kitti_{seq}_N{N}_best_frames.pkl.gz — L best frame pairs (subsampled keypoints)

Usage:
    python3 scripts/run_kitti_subsample.py -N 500
    python3 scripts/run_kitti_subsample.py -N 500 -K 500 -L 5
    python3 scripts/run_kitti_subsample.py -N 500 --seqs 0 5 9
"""

import sys
import argparse
import subprocess
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

FRAMES_DIR = project_root / "assets/outputs/kitti/both"
GT_DIR     = project_root / "assets/trajectories/kitti"
KITTI_DATA = Path("/home/joe/data/kitti/color/dataset/sequences")
OUT_DIR    = project_root / "assets/outputs/subsample_errors/kitti"

FRAME_COUNTS = {
    0: 4541, 1: 1101, 2: 4661, 3:  801, 4:  271,
    5: 2761, 6: 1101, 7: 1101, 8: 4071, 9: 1591, 10: 1201,
}
DEFAULT_SEQS = list(range(11))


def build_parser():
    p = argparse.ArgumentParser(prog="run_kitti_subsample")
    p.add_argument("-N", type=int, required=True, dest="N",
                   help="Keypoints to subsample per pair")
    p.add_argument("-K", type=int, default=1000, dest="K",
                   help="Random trials per frame pair (default: 1000)")
    p.add_argument("-L", type=int, default=1, dest="L",
                   help="Best frame pairs to save per trial (default: 1)")
    p.add_argument("--seqs", type=int, nargs="+", default=DEFAULT_SEQS,
                   help=f"Sequences to process (default: {DEFAULT_SEQS})")
    p.add_argument("--seed", type=int, default=None)
    return p


def main():
    args = build_parser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"KITTI subsample evaluation  N={args.N}  K={args.K}  L={args.L}")
    print(f"Sequences: {args.seqs}")

    for seq in args.seqs:
        seq_str     = f"{seq:02d}"
        n_frames    = FRAME_COUNTS[seq]
        frames_path = FRAMES_DIR / f"kitti_{seq_str}_both_frames.pkl.gz"
        gt_path     = GT_DIR     / f"{seq_str}.txt"
        calib_path  = KITTI_DATA / seq_str / "calib.txt"
        out_frames  = OUT_DIR / f"kitti_{seq_str}_N{args.N}_best_frames.pkl.gz"
        out_errors  = OUT_DIR / f"kitti_{seq_str}_N{args.N}_errors.npz"

        print(f"\n── seq {seq_str}  ({n_frames} frames) ──────────────────────────────")

        for label, path in [("frames", frames_path), ("GT", gt_path), ("calib", calib_path)]:
            if not path.exists():
                print(f"  [SKIP] {label} not found: {path}")
                break
        else:
            cmd = [
                sys.executable, str(project_root / "scripts/subsample_pose_eval.py"),
                str(frames_path), str(gt_path), str(out_frames),
                "--start", "0", "--end", str(n_frames - 1),
                "-N", str(args.N), "-K", str(args.K), "-L", str(args.L),
                "--config_path", str(calib_path), "--camera", "kitti",
                "--gt_format", "kitti",
                "--errors_out", str(out_errors),
            ]
            if args.seed is not None:
                cmd += ["--seed", str(args.seed)]

            subprocess.run(cmd, check=True)
            print(f"  errors → {out_errors}")
            print(f"  frames → {out_frames}")

    print(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
