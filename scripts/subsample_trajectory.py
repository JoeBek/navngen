#!/usr/bin/env python3
"""
subsample_trajectory.py

Runs the subsampling pose estimator over a frame sequence and saves the
accumulated trajectory to a text file.

For each consecutive pair, K random subsamples of N keypoints are tried.
The trial with the most RANSAC inliers is kept.  Falls back to all matched
keypoints if every trial fails.

Usage:
    python3 subsample_trajectory.py <frames.pkl.gz> <output.txt> \\
        -N N -K K \\
        --config_path <yaml> --camera surfnav|kitti|euroc \\
        [--start S] [--end E] \\
        [--traj_format tum|kitti] [--ts_scale FLOAT] [--seed INT]
"""

import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.navngen.export_trajectory import load_frames
from src.navngen.trajectory import Solver, compose_with_unit_direction


# ─────────────────────────── trajectory export ───────────────────────────────

def save_tum(poses, frames, indices, ts_scale, path):
    """poses: list of (R, t) in world frame, one per index in indices."""
    with open(path, "w") as f:
        for (R, t), fi in zip(poses, indices):
            ts = float(getattr(frames[fi], "timestamp", fi)) * ts_scale
            qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
            tx, ty, tz = t
            f.write(f"{ts:.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
                    f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")


def save_kitti(poses, path):
    """poses: list of (R, t) in world frame."""
    with open(path, "w") as f:
        for R, t in poses:
            row = np.hstack([R, t.reshape(3, 1)]).flatten()
            f.write(" ".join(f"{v:.9f}" for v in row) + "\n")


# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="subsample_trajectory",
        description="Accumulate a trajectory using subsampled keypoint estimates.",
    )
    p.add_argument("frames",  type=Path, help="Input .pkl.gz frame file")
    p.add_argument("output",  type=Path, help="Output trajectory text file")
    p.add_argument("-N",      type=int, required=True, dest="N",
                   help="Keypoints to subsample per trial")
    p.add_argument("-K",      type=int, required=True, dest="K",
                   help="Trials per frame pair")
    p.add_argument("--config_path", type=Path, required=True,
                   help="Camera calibration YAML")
    p.add_argument("--camera", default="surfnav",
                   choices=["surfnav", "euroc", "kitti"],
                   help="Camera config format (default: surfnav)")
    p.add_argument("--start", type=int, default=None,
                   help="First frame index (default: 0)")
    p.add_argument("--end",   type=int, default=None,
                   help="Last frame index inclusive (default: last)")
    p.add_argument("--traj_format", default="tum", choices=["tum", "kitti"],
                   help="Output trajectory format (default: tum)")
    p.add_argument("--ts_scale", type=float, default=1.0,
                   help="Multiply frame.timestamp by this to get seconds "
                        "(use 1e-9 for nanosecond timestamps; TUM only)")
    p.add_argument("--seed", type=int, default=None)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print(f"Loading frames: {args.frames}")
    frames = load_frames(args.frames)
    print(f"  {len(frames)} frames")

    solver = Solver(args.config_path, config_type=args.camera)

    start = args.start if args.start is not None else 0
    end   = args.end   if args.end   is not None else len(frames) - 1

    if start < 0 or end >= len(frames) or start >= end:
        print(f"Error: invalid range [{start}, {end}] for {len(frames)} frames")
        sys.exit(1)

    # ── accumulate trajectory ─────────────────────────────────────────────────
    R_world = np.eye(3)
    t_world = np.zeros(3)

    pose_list  = [(R_world.copy(), t_world.copy())]   # pose for frame `start`
    frame_list = [start]
    n_fallback = 0
    n_failed   = 0

    for i in tqdm(range(start, end), desc="Building trajectory"):
        j = i + 1
        f_prev = frames[i]
        f_curr = frames[j]

        if f_curr.matches is None or f_prev.kpts is None or f_curr.kpts is None:
            tqdm.write(f"  [SKIP] pair ({i},{j}): missing matches/keypoints — holding pose")
            pose_list.append((R_world.copy(), t_world.copy()))
            frame_list.append(j)
            n_failed += 1
            continue

        kpts0 = f_prev.kpts
        kpts1 = f_curr.kpts
        if hasattr(kpts0, "numpy"):
            kpts0 = kpts0.numpy()
        if hasattr(kpts1, "numpy"):
            kpts1 = kpts1.numpy()
        kpts0 = np.asarray(kpts0)
        kpts1 = np.asarray(kpts1)

        matches = f_curr.matches
        if hasattr(matches, "numpy"):
            matches = matches.numpy()
        matches = np.asarray(matches)

        n_matches = len(matches)
        n_sample  = min(args.N, n_matches)

        best_inliers = -1
        best_R = best_t = None

        if n_sample >= 5:
            for _ in range(args.K):
                idx = np.random.choice(n_matches, size=n_sample, replace=False)
                sub = matches[idx]
                m0  = kpts0[sub[:, 0]]
                m1  = kpts1[sub[:, 1]]

                try:
                    pose, info = solver.solve_relative_pose(m0, m1)
                    n_inliers  = int(np.sum(info["inliers"]))
                except Exception:
                    continue

                if n_inliers > best_inliers:
                    best_inliers = n_inliers
                    best_R = pose.R.copy()
                    best_t = pose.t.copy()

        # fallback: use all matches
        if best_R is None:
            all_m0 = kpts0[matches[:, 0]]
            all_m1 = kpts1[matches[:, 1]]
            try:
                pose, _ = solver.solve_relative_pose(all_m0, all_m1)
                best_R, best_t = pose.R.copy(), pose.t.copy()
                n_fallback += 1
                tqdm.write(f"  [FALLBACK] pair ({i},{j}): all K trials failed, used full match set")
            except Exception:
                tqdm.write(f"  [SKIP] pair ({i},{j}): fallback also failed — holding pose")
                pose_list.append((R_world.copy(), t_world.copy()))
                frame_list.append(j)
                n_failed += 1
                continue

        R_world, t_world = compose_with_unit_direction(R_world, t_world, best_R, best_t)
        pose_list.append((R_world.copy(), t_world.copy()))
        frame_list.append(j)

    print(f"\n{len(pose_list)} poses accumulated  "
          f"({n_fallback} fallbacks, {n_failed} held)")

    # ── save ──────────────────────────────────────────────────────────────────
    print(f"Saving {args.traj_format} trajectory → {args.output}")
    if args.traj_format == "tum":
        save_tum(pose_list, frames, frame_list, args.ts_scale, args.output)
    else:
        save_kitti(pose_list, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
