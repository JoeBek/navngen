#!/usr/bin/env python3
"""
eval_inliers.py

For each consecutive frame pair, take the inlier mask stored in
``frame.info['inliers']`` from a previous pose estimation, re-solve the
relative pose using only those inliers (the full set, no subsampling), and
score against ground truth.

Errors are reported in degrees so the resulting .npz plays nicely with
plot_auc.py and plot_subsample_errors.py:
  rot_errors    geodesic rotation error (degrees)
  trans_errors  angle between unit translation vectors (degrees)

Usage:
    python3 eval_inliers.py <frames.pkl.gz> <gt_file> <output.npz> \\
        --config_path <yaml> --camera surfnav|kitti|euroc \\
        [--gt_format kitti|tum] [--ts_scale FLOAT] [--start S] [--end E]
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
from src.navngen.trajectory import Solver


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


def rotation_error_deg(R_est, R_gt):
    dR = R_est.T @ R_gt
    cos_val = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def translation_error_deg(t_est, t_gt):
    ne, ng = np.linalg.norm(t_est), np.linalg.norm(t_gt)
    if ne < 1e-9 or ng < 1e-9:
        return float('nan')
    cos_val = np.clip(np.dot(t_est / ne, t_gt / ng), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def print_stats(name, vals):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    if not len(v):
        print(f"  {name}: no valid data")
        return
    pcts = [50, 75, 80, 90, 95, 99]
    pv = np.percentile(v, pcts)
    print(f"\n  {name}  [deg]  n={len(v)}")
    print(f"    mean={np.mean(v):.4f}  median={np.median(v):.4f}  std={np.std(v):.4f}")
    print(f"    min ={np.min(v):.4f}   max   ={np.max(v):.4f}")
    for p, x in zip(pcts, pv):
        print(f"    p{p:<3} ={x:.4f}")


def build_parser():
    p = argparse.ArgumentParser(prog="eval_inliers")
    p.add_argument("frames",      type=Path)
    p.add_argument("gt",          type=Path)
    p.add_argument("output",      type=Path, help=".npz output")
    p.add_argument("--config_path", type=Path, required=True)
    p.add_argument("--camera",    default="surfnav",
                   choices=["surfnav", "euroc", "kitti"])
    p.add_argument("--gt_format", default="tum", choices=["kitti", "tum"])
    p.add_argument("--ts_scale",  type=float, default=1.0)
    p.add_argument("--start",     type=int, default=None)
    p.add_argument("--end",       type=int, default=None)
    return p


def main():
    args = build_parser().parse_args()

    print(f"Loading frames: {args.frames}")
    frames = load_frames(args.frames)
    print(f"  {len(frames)} frames")

    print(f"Loading GT ({args.gt_format}): {args.gt}")
    gt_data = (load_gt_kitti(args.gt) if args.gt_format == "kitti"
               else load_gt_tum(args.gt))
    print(f"  {len(gt_data)} GT entries")

    solver = Solver(args.config_path, config_type=args.camera)

    start = args.start if args.start is not None else 0
    end   = args.end   if args.end   is not None else len(frames) - 1

    frame_range = range(start, end + 1)
    tum_map = (build_tum_frame_map(gt_data, frames, frame_range, args.ts_scale)
               if args.gt_format == "tum" else None)

    def get_gt_relative(i, j):
        if args.gt_format == "kitti":
            if i >= len(gt_data) or j >= len(gt_data):
                return None, None
            return kitti_relative_pose(gt_data, i, j)
        gi, gj = tum_map.get(i), tum_map.get(j)
        if gi is None or gj is None:
            return None, None
        return tum_relative_pose(gt_data, gi, gj)

    pair_indices  = []
    trans_errors  = []
    rot_errors    = []
    n_inliers_arr = []
    n_matches_arr = []
    n_skipped     = 0
    skip_reasons  = {"missing": 0, "no_inliers": 0, "few_inliers": 0,
                     "no_gt": 0, "solver_failed": 0}

    for i in tqdm(range(start, end), desc="Evaluating pairs"):
        j = i + 1
        f_prev, f_curr = frames[i], frames[j]

        if (f_curr.matches is None or f_prev.kpts is None
                or f_curr.kpts is None or f_curr.info is None):
            n_skipped += 1
            skip_reasons["missing"] += 1
            continue

        inliers = f_curr.info.get("inliers")
        if inliers is None:
            n_skipped += 1
            skip_reasons["no_inliers"] += 1
            continue
        inliers = np.asarray(inliers, dtype=bool)

        kpts0   = np.asarray(f_prev.kpts.numpy() if hasattr(f_prev.kpts, 'numpy') else f_prev.kpts)
        kpts1   = np.asarray(f_curr.kpts.numpy() if hasattr(f_curr.kpts, 'numpy') else f_curr.kpts)
        matches = np.asarray(f_curr.matches.numpy() if hasattr(f_curr.matches, 'numpy') else f_curr.matches)

        if len(inliers) != len(matches):
            n_skipped += 1
            skip_reasons["missing"] += 1
            continue

        sub = matches[inliers]
        if len(sub) < 5:
            n_skipped += 1
            skip_reasons["few_inliers"] += 1
            continue

        R_gt, t_gt = get_gt_relative(i, j)
        if R_gt is None or np.linalg.norm(t_gt) < 1e-9:
            n_skipped += 1
            skip_reasons["no_gt"] += 1
            continue

        m0 = kpts0[sub[:, 0]]
        m1 = kpts1[sub[:, 1]]
        try:
            pose, _ = solver.solve_relative_pose(m0, m1)
            te = translation_error_deg(pose.t, t_gt)
            re = rotation_error_deg(pose.R, R_gt)
        except Exception:
            n_skipped += 1
            skip_reasons["solver_failed"] += 1
            continue

        pair_indices.append(i)
        trans_errors.append(te)
        rot_errors.append(re)
        n_inliers_arr.append(int(inliers.sum()))
        n_matches_arr.append(int(len(matches)))

    print(f"\n{len(pair_indices)} valid pairs, {n_skipped} skipped")
    for k, v in skip_reasons.items():
        if v:
            print(f"  {k}: {v}")
    print_stats("translation error (deg)", trans_errors)
    print_stats("rotation error (deg)",    rot_errors)

    np.savez(
        args.output,
        pair_indices = np.array(pair_indices, dtype=np.int32),
        trans_errors = np.array(trans_errors),
        rot_errors   = np.array(rot_errors),
        n_inliers    = np.array(n_inliers_arr, dtype=np.int32),
        n_matches    = np.array(n_matches_arr, dtype=np.int32),
    )
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
