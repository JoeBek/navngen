#!/usr/bin/env python3
"""
subsample.py

Subsample keypoint matches K times per frame pair and evaluate against GT.

Two modes:
  --mode errors   save per-pair trial error distributions to a .npz file
  --mode frames   save the L best-scoring frame pairs to a .pkl.gz file

Error metrics (--metric):
  translation     L2 distance between unit translation vectors  [0, 2]
  rotation        geodesic angle between rotation matrices (degrees)

Restrict to a single component:
  --axis x|y|z    use only that component of the unit translation vectors
  --angle yaw|pitch|roll   use only that Euler angle component (degrees, YXZ)

Usage:
    python3 subsample.py <frames.pkl.gz> <gt_file> <output> \\
        --start S --end E -N N -K K \\
        --config_path <yaml> --camera surfnav|kitti|euroc \\
        --mode errors|frames \\
        [--metric translation|rotation] \\
        [--axis x|y|z] [--angle yaw|pitch|roll] \\
        [-L L]  (required for --mode frames) \\
        [--gt_format kitti|tum] [--ts_scale FLOAT] [--seed INT]
"""

import sys
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.navngen.export_trajectory import load_frames, export_frames
from src.navngen.trajectory import Solver


# ─────────────────────────── GT loading ──────────────────────────────────────

def load_gt_kitti(path: Path):
    poses = []
    with open(path) as f:
        for line in f:
            vals = [float(v) for v in line.strip().split()]
            P = np.array(vals).reshape(3, 4)
            poses.append(np.vstack([P, [0, 0, 0, 1]]))
    return poses


def load_gt_tum(path: Path):
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


# ─────────────────────────── error metrics ───────────────────────────────────

AXIS_IDX  = {'x': 0, 'y': 1, 'z': 2}
ANGLE_IDX = {'yaw': 0, 'pitch': 1, 'roll': 2}   # YXZ Euler decomposition


def compute_error(R_est, t_est, R_gt, t_gt, metric, axis, angle):
    """Return a scalar error value for one trial."""
    if metric == 'translation':
        ne = np.linalg.norm(t_est)
        ng = np.linalg.norm(t_gt)
        if ne < 1e-9 or ng < 1e-9:
            return float('nan')
        u_est = t_est / ne
        u_gt  = t_gt  / ng
        if axis is not None:
            return float(abs(u_est[AXIS_IDX[axis]] - u_gt[AXIS_IDX[axis]]))
        return float(np.linalg.norm(u_est - u_gt))

    else:  # rotation
        if angle is not None:
            euler_est = Rotation.from_matrix(R_est).as_euler('YXZ', degrees=True)
            euler_gt  = Rotation.from_matrix(R_gt ).as_euler('YXZ', degrees=True)
            diff = abs(euler_est[ANGLE_IDX[angle]] - euler_gt[ANGLE_IDX[angle]])
            if diff > 180:
                diff = 360 - diff
            return float(diff)
        dR      = R_est.T @ R_gt
        cos_val = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_val)))


def error_label(metric, axis, angle):
    if metric == 'translation':
        suffix = f" (axis={axis})" if axis else " (L2)"
        return f"unit translation error{suffix}"
    suffix = f" ({angle})" if angle else " (geodesic °)"
    return f"rotation error{suffix}"


# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(prog="subsample")
    p.add_argument("frames",  type=Path)
    p.add_argument("gt",      type=Path)
    p.add_argument("output",  type=Path,
                   help=".npz for --mode errors, .pkl.gz for --mode frames")
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end",   type=int, required=True)
    p.add_argument("-N",      type=int, required=True, dest="N")
    p.add_argument("-K",      type=int, required=True, dest="K")
    p.add_argument("--config_path", type=Path, required=True)
    p.add_argument("--camera", default="surfnav",
                   choices=["surfnav", "euroc", "kitti"])
    p.add_argument("--mode",   required=True, choices=["errors", "frames"])
    p.add_argument("--metric", default="translation",
                   choices=["translation", "rotation"])
    p.add_argument("--axis",  default=None, choices=["x", "y", "z"],
                   help="Restrict translation error to one axis of the unit vector")
    p.add_argument("--angle", default=None, choices=["yaw", "pitch", "roll"],
                   help="Restrict rotation error to one Euler component (YXZ)")
    p.add_argument("-L",      type=int, default=None, dest="L",
                   help="Number of best pairs to keep (required for --mode frames)")
    p.add_argument("--gt_format", default="kitti", choices=["kitti", "tum"])
    p.add_argument("--ts_scale",  type=float, default=1.0)
    p.add_argument("--seed",      type=int, default=None)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "frames" and args.L is None:
        parser.error("--mode frames requires -L")
    if args.axis is not None and args.metric != "translation":
        parser.error("--axis requires --metric translation")
    if args.angle is not None and args.metric != "rotation":
        parser.error("--angle requires --metric rotation")

    if args.seed is not None:
        np.random.seed(args.seed)

    print(f"Loading frames : {args.frames}")
    frames = load_frames(args.frames)
    print(f"  {len(frames)} frames")

    print(f"Loading GT ({args.gt_format}): {args.gt}")
    gt_data = load_gt_kitti(args.gt) if args.gt_format == "kitti" else load_gt_tum(args.gt)
    print(f"  {len(gt_data)} GT entries")

    solver = Solver(args.config_path, config_type=args.camera)

    start, end = args.start, args.end
    if start < 0 or end >= len(frames) or start >= end:
        print(f"Error: invalid range [{start}, {end}] for {len(frames)} frames")
        sys.exit(1)

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

    elabel = error_label(args.metric, args.axis, args.angle)
    print(f"Metric: {elabel}  |  N={args.N}  K={args.K}")

    # ── per-pair evaluation ───────────────────────────────────────────────────
    pair_results = []
    n_skipped = 0

    for i in tqdm(range(start, end), desc="Evaluating pairs"):
        j = i + 1
        f_prev, f_curr = frames[i], frames[j]

        if f_curr.matches is None or f_prev.kpts is None or f_curr.kpts is None:
            n_skipped += 1
            continue

        kpts0 = np.asarray(f_prev.kpts.numpy() if hasattr(f_prev.kpts, 'numpy') else f_prev.kpts)
        kpts1 = np.asarray(f_curr.kpts.numpy() if hasattr(f_curr.kpts, 'numpy') else f_curr.kpts)
        matches = np.asarray(f_curr.matches.numpy() if hasattr(f_curr.matches, 'numpy') else f_curr.matches)

        n_matches = len(matches)
        n_sample  = min(args.N, n_matches)

        if n_sample < 5:
            n_skipped += 1
            continue

        R_gt, t_gt = get_gt_relative(i, j)
        if R_gt is None or np.linalg.norm(t_gt) < 1e-9:
            n_skipped += 1
            continue

        all_m0 = kpts0[matches[:, 0]]
        all_m1 = kpts1[matches[:, 1]]

        # baseline
        base_err = float('nan')
        try:
            pose_b, _ = solver.solve_relative_pose(all_m0, all_m1)
            base_err  = compute_error(pose_b.R, pose_b.t, R_gt, t_gt,
                                      args.metric, args.axis, args.angle)
        except Exception:
            pass

        # K trials — NaN-padded so arrays are always length K
        tc = np.full(args.K, float('nan'))
        best_err = np.inf
        best_R = best_t = best_sub = best_info = None

        for k in range(args.K):
            idx = np.random.choice(n_matches, size=n_sample, replace=False)
            sub = matches[idx]
            try:
                pose, info = solver.solve_relative_pose(kpts0[sub[:, 0]], kpts1[sub[:, 1]])
                err = compute_error(pose.R, pose.t, R_gt, t_gt,
                                    args.metric, args.axis, args.angle)
            except Exception:
                continue

            tc[k] = err
            if err < best_err:
                best_err  = err
                best_R    = pose.R.copy()
                best_t    = pose.t.copy()
                best_sub  = sub.copy()
                best_info = info

        if best_R is None:
            tqdm.write(f"  [SKIP] pair ({i},{j}): all {args.K} trials failed")
            n_skipped += 1
            continue

        pair_results.append(dict(
            frame_i=i,
            R_best=best_R, t_best=best_t, best_sub=best_sub, best_info=best_info,
            best_err=best_err,
            median_err=float(np.nanmedian(tc)),
            worst_err=float(np.nanmax(tc)),
            base_err=base_err,
            n_matches=n_matches,
            n_trials_ok=int(np.sum(np.isfinite(tc))),
            trial_errors=tc,
        ))

    print(f"\n{len(pair_results)} valid pairs, {n_skipped} skipped")
    if not pair_results:
        sys.exit(1)

    # ── stats ─────────────────────────────────────────────────────────────────
    def _s(vals):
        v = np.asarray([x for x in vals if np.isfinite(x)])
        if not len(v):
            return "n/a"
        return f"mean={np.mean(v):.4f}  median={np.median(v):.4f}  std={np.std(v):.4f}  min={np.min(v):.4f}  max={np.max(v):.4f}"

    print()
    print(f"Metric: {elabel}")
    print(f"  subsampled (best of K)  {_s(r['best_err']  for r in pair_results)}")
    print(f"  subsampled (median)     {_s(r['median_err'] for r in pair_results)}")
    print(f"  baseline (all matches)  {_s(r['base_err']  for r in pair_results)}")

    # ── output ────────────────────────────────────────────────────────────────
    if args.mode == "errors":
        M = len(pair_results)
        np.savez(
            args.output,
            pair_indices  = np.array([r["frame_i"]     for r in pair_results], dtype=np.int32),
            trial_errors  = np.stack([r["trial_errors"] for r in pair_results]),   # [M, K]
            base_errors   = np.array([r["base_err"]     for r in pair_results]),
            N=np.int32(args.N), K=np.int32(args.K),
            metric=np.bytes_(args.metric),
            axis =np.bytes_(args.axis  or ""),
            angle=np.bytes_(args.angle or ""),
        )
        print(f"\nError arrays ({M}×{args.K}) → {args.output}")

    else:  # frames
        pair_results.sort(key=lambda r: r["best_err"])
        L = min(args.L, len(pair_results))
        selected = sorted(pair_results[:L], key=lambda r: r["frame_i"])

        print(f"\n{'pair':>10}  {'best':>10}  {'median':>10}  {'baseline':>10}  {'N_matches':>10}  {'K_ok':>5}")
        print("  " + "-" * 60)
        for r in selected:
            i = r["frame_i"]
            print(f"  ({i:4d},{i+1:4d})  {r['best_err']:10.4f}  {r['median_err']:10.4f}  "
                  f"{r['base_err']:10.4f}  {r['n_matches']:10d}  {r['n_trials_ok']:5d}")

        out_frames = []
        for r in selected:
            i, j   = r["frame_i"], r["frame_i"] + 1
            sub     = r["best_sub"]
            N_sub   = len(sub)

            f_prev = copy.copy(frames[i])
            orig_i = frames[i].kpts
            f_prev.kpts = (orig_i[torch.from_numpy(sub[:, 0]).long()]
                           if isinstance(orig_i, torch.Tensor)
                           else np.asarray(orig_i)[sub[:, 0]])
            f_prev.features = None
            f_prev.matches  = None

            f_curr = copy.copy(frames[j])
            orig_j = frames[j].kpts
            f_curr.kpts = (orig_j[torch.from_numpy(sub[:, 1]).long()]
                           if isinstance(orig_j, torch.Tensor)
                           else np.asarray(orig_j)[sub[:, 1]])
            f_curr.features = None

            identity = np.stack([np.arange(N_sub), np.arange(N_sub)], axis=1)
            orig_m   = frames[j].matches
            f_curr.matches = (torch.tensor(identity, dtype=orig_m.dtype)
                              if isinstance(orig_m, torch.Tensor) else identity)

            n = np.linalg.norm(r["t_best"])
            f_curr.E    = (r["R_best"], r["t_best"] / n if n > 1e-9 else r["t_best"])
            f_curr.info = r["best_info"]

            out_frames.extend([f_prev, f_curr])

        print(f"\nframetool index map")
        print(f"  {'prev':>6}  {'curr':>6}  {'original pair':>14}  {'best err':>10}")
        print(f"  {'-'*6}  {'-'*6}  {'-'*14}  {'-'*10}")
        for k, r in enumerate(selected):
            i = r["frame_i"]
            print(f"  {2*k:>6}  {2*k+1:>6}  ({i:4d}, {i+1:4d})      {r['best_err']:10.4f}")

        export_frames(out_frames, args.output)
        print(f"\n{len(out_frames)} frames → {args.output}")

    print("Done.")


if __name__ == "__main__":
    main()
