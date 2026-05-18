#!/usr/bin/env python3
"""
subsample_trajectory.py

Accumulate a trajectory by running K random subsamples of N keypoints per pair.

Selection criterion (per pair):
  default          most RANSAC inliers  (no GT needed)
  --gt <file>      lowest translation error vs ground truth  (oracle mode)

Usage:
    python3 subsample_trajectory.py <frames.pkl.gz> <output.txt> \\
        -N N -K K \\
        --config_path <yaml> --camera surfnav|kitti|euroc \\
        [--start S] [--end E] \\
        [--traj_format tum|kitti] [--ts_scale FLOAT] [--seed INT] \\
        [--gt <gt_file> --gt_format kitti|tum]
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


# ─────────────────────────── GT helpers ──────────────────────────────────────

def load_gt_kitti(path):
    poses = []
    with open(path) as f:
        for line in f:
            vals = [float(v) for v in line.strip().split()]
            P = np.array(vals).reshape(3, 4)
            poses.append(np.vstack([P, [0, 0, 0, 1]]))
    return poses


def load_gt_tum(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = list(map(float, line.split()))
            if len(parts) != 8:
                continue
            ts, tx, ty, tz, qx, qy, qz, qw = parts
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            entries.append((ts, R, np.array([tx, ty, tz])))
    return entries


def build_tum_frame_map(gt_entries, frames, frame_range, ts_scale):
    gt_ts = np.array([e[0] for e in gt_entries])
    mapping = {}
    for i in frame_range:
        ts = getattr(frames[i], 'timestamp', None)
        if ts is None:
            continue
        mapping[i] = int(np.argmin(np.abs(gt_ts - float(ts) * ts_scale)))
    return mapping


def kitti_relative_pose(poses, i, j):
    rel = np.linalg.inv(poses[j]) @ poses[i]
    return rel[:3, :3], rel[:3, 3]


def tum_relative_pose(entries, gi, gj):
    _, R0, t0 = entries[gi]
    _, R1, t1 = entries[gj]
    return R0.T @ R1, R0.T @ (t1 - t0)


def translation_error(t_est, t_gt):
    ne, ng = np.linalg.norm(t_est), np.linalg.norm(t_gt)
    if ne < 1e-9 or ng < 1e-9:
        return float('inf')
    return float(np.linalg.norm(t_est / ne - t_gt / ng))


def rotation_error(R_est, R_gt):
    dR = R_est.T @ R_gt
    cos_val = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


# ─────────────────────────── trajectory export ───────────────────────────────

def save_tum(poses, frames, indices, ts_scale, path):
    with open(path, "w") as f:
        for (R, t), fi in zip(poses, indices):
            ts = float(getattr(frames[fi], "timestamp", fi)) * ts_scale
            qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
            tx, ty, tz = t
            f.write(f"{ts:.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
                    f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")


def save_kitti(poses, path):
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
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end",   type=int, default=None)
    p.add_argument("--traj_format", default="tum", choices=["tum", "kitti"],
                   help="Output trajectory format (default: tum)")
    p.add_argument("--ts_scale", type=float, default=1.0,
                   help="Multiply frame.timestamp by this to get seconds "
                        "(use 1e-9 for nanosecond timestamps; TUM only)")
    p.add_argument("--seed", type=int, default=None)
    # oracle GT selection
    p.add_argument("--gt", type=Path, default=None,
                   help="GT file — enables oracle mode: pick trial with lowest "
                        "error instead of most inliers")
    p.add_argument("--gt_format", default="tum", choices=["kitti", "tum"])
    p.add_argument("--metric", default="translation", choices=["translation", "rotation"],
                   help="Error metric for oracle selection (default: translation)")
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

    # ── GT setup (oracle mode) ────────────────────────────────────────────────
    gt_data  = None
    tum_map  = None
    oracle   = args.gt is not None

    if oracle:
        print(f"Loading GT ({args.gt_format}): {args.gt}")
        gt_data = (load_gt_kitti(args.gt) if args.gt_format == "kitti"
                   else load_gt_tum(args.gt))
        print(f"  {len(gt_data)} GT entries  [oracle mode: select by {args.metric} error]")
        if args.gt_format == "tum":
            tum_map = build_tum_frame_map(gt_data, frames,
                                          range(start, end + 1), args.ts_scale)
    else:
        print("Selection: most RANSAC inliers")

    def get_gt_relative(i, j):
        if args.gt_format == "kitti":
            if i >= len(gt_data) or j >= len(gt_data):
                return None, None
            return kitti_relative_pose(gt_data, i, j)
        gi, gj = tum_map.get(i), tum_map.get(j)
        if gi is None or gj is None:
            return None, None
        return tum_relative_pose(gt_data, gi, gj)

    # ── accumulate trajectory ─────────────────────────────────────────────────
    R_world = np.eye(3)
    t_world = np.zeros(3)

    pose_list  = [(R_world.copy(), t_world.copy())]
    frame_list = [start]
    n_fallback = 0
    n_failed   = 0

    for i in tqdm(range(start, end), desc="Building trajectory"):
        j = i + 1
        f_prev, f_curr = frames[i], frames[j]

        if f_curr.matches is None or f_prev.kpts is None or f_curr.kpts is None:
            tqdm.write(f"  [SKIP] pair ({i},{j}): missing matches/keypoints — holding pose")
            pose_list.append((R_world.copy(), t_world.copy()))
            frame_list.append(j)
            n_failed += 1
            continue

        kpts0 = np.asarray(f_prev.kpts.numpy() if hasattr(f_prev.kpts, 'numpy') else f_prev.kpts)
        kpts1 = np.asarray(f_curr.kpts.numpy() if hasattr(f_curr.kpts, 'numpy') else f_curr.kpts)
        matches = np.asarray(f_curr.matches.numpy() if hasattr(f_curr.matches, 'numpy') else f_curr.matches)

        n_matches = len(matches)
        n_sample  = min(args.N, n_matches)

        best_score = float('inf') if oracle else -1
        best_R = best_t = None

        if n_sample >= 5:
            # resolve GT for this pair once (oracle mode only)
            if oracle:
                R_gt, t_gt = get_gt_relative(i, j)
                gt_valid = R_gt is not None and np.linalg.norm(t_gt) > 1e-9
            else:
                gt_valid = False

            for _ in range(args.K):
                idx = np.random.choice(n_matches, size=n_sample, replace=False)
                sub = matches[idx]
                try:
                    pose, info = solver.solve_relative_pose(
                        kpts0[sub[:, 0]], kpts1[sub[:, 1]])
                except Exception:
                    continue

                if oracle and gt_valid:
                    score = (translation_error(pose.t, t_gt)
                             if args.metric == "translation"
                             else rotation_error(pose.R, R_gt))
                    if score < best_score:
                        best_score = score
                        best_R = pose.R.copy()
                        best_t = pose.t.copy()
                else:
                    n_inliers = int(np.sum(info["inliers"]))
                    if n_inliers > best_score:
                        best_score = n_inliers
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

    print(f"Saving {args.traj_format} trajectory → {args.output}")
    if args.traj_format == "tum":
        save_tum(pose_list, frames, frame_list, args.ts_scale, args.output)
    else:
        save_kitti(pose_list, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
