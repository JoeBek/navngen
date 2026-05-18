#!/usr/bin/env python3
"""
subsample_pose_eval.py

For each consecutive frame pair in a given index range, randomly subsample N
keypoint matches K times, estimate the relative pose each time, and score each
trial against ground truth.  The L frame pairs with the lowest minimum-over-K
combined pose error are written to a new pickle.

Usage:
    python3 subsample_pose_eval.py <frames.pkl.gz> <gt_file> <output.pkl.gz> \\
        --start S --end E -N N -K K -L L \\
        --config_path <yaml> [--camera surfnav|euroc|kitti] \\
        [--gt_format kitti|tum] [--ts_scale FLOAT] [--seed INT]

GT formats:
    kitti  — one "r11 r12 ... tz" (12-value) line per frame, 0-indexed
    tum    — "timestamp tx ty tz qx qy qz qw" lines, matched by timestamp
             (set --ts_scale 1e-9 when frame timestamps are nanoseconds)
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
    """List of 4×4 numpy arrays (camera-to-world), one per frame."""
    poses = []
    with open(path) as f:
        for line in f:
            vals = [float(v) for v in line.strip().split()]
            P = np.array(vals).reshape(3, 4)
            poses.append(np.vstack([P, [0, 0, 0, 1]]))
    return poses


def load_gt_tum(path: Path):
    """List of (timestamp_sec, R_c2w 3×3, t_world 3) tuples."""
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
    """
    Map each frame index in frame_range to the closest GT entry by timestamp.
    ts_scale converts frame.timestamp to seconds (e.g. 1e-9 for nanoseconds).
    Returns dict {frame_idx: gt_entry_idx}.
    """
    gt_ts = np.array([e[0] for e in gt_entries])
    mapping = {}
    for i in frame_range:
        ts = getattr(frames[i], 'timestamp', None)
        if ts is None:
            continue
        mapping[i] = int(np.argmin(np.abs(gt_ts - float(ts) * ts_scale)))
    return mapping


# ─────────────────────────── relative pose helpers ───────────────────────────

def kitti_relative_pose(poses, i, j):
    rel = np.linalg.inv(poses[j]) @ poses[i]
    return rel[:3, :3], rel[:3, 3]


def tum_relative_pose(entries, gi, gj):
    _, R0, t0 = entries[gi]
    _, R1, t1 = entries[gj]
    R_rel = R0.T @ R1
    t_rel = R0.T @ (t1 - t0)
    return R_rel, t_rel


# ─────────────────────────── error metrics ───────────────────────────────────

def rotation_error_deg(R_est, R_gt):
    dR = R_est.T @ R_gt
    cos_val = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def translation_error_deg(t_est, t_gt):
    ne, ng = np.linalg.norm(t_est), np.linalg.norm(t_gt)
    if ne < 1e-9 or ng < 1e-9:
        return 180.0
    cos_val = np.clip(np.dot(t_est / ne, t_gt / ng), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="subsample_pose_eval",
        description="Subsample keypoints, re-estimate relative poses, save L best pairs.",
    )
    p.add_argument("frames",  type=Path, help="Input .pkl.gz frame file")
    p.add_argument("gt",      type=Path, help="Ground-truth pose file")
    p.add_argument("output",  type=Path, help="Output .pkl.gz file")
    p.add_argument("--start", type=int, required=True,  help="First frame index (inclusive)")
    p.add_argument("--end",   type=int, required=True,  help="Last frame index (inclusive)")
    p.add_argument("-N",      type=int, required=True,  dest="N",
                   help="Number of keypoints to subsample per trial")
    p.add_argument("-K",      type=int, required=True,  dest="K",
                   help="Number of random trials per frame pair")
    p.add_argument("-L",      type=int, required=True,  dest="L",
                   help="Number of best frame pairs to keep")
    p.add_argument("--config_path", type=Path, required=True,
                   help="Path to the camera calibration file")
    p.add_argument("--camera", default="surfnav",
                   choices=["surfnav", "euroc", "kitti"],
                   help="Camera config format (default: surfnav)")
    p.add_argument("--gt_format",   default="kitti",
                   choices=["kitti", "tum"],
                   help="Ground-truth file format (default: kitti)")
    p.add_argument("--ts_scale",    type=float, default=1.0,
                   help="Multiply frame.timestamp by this to get seconds "
                        "(use 1e-9 for nanosecond timestamps; TUM only)")
    p.add_argument("--errors_out", type=Path, default=None, metavar="PATH",
                   help="Save per-pair trial error arrays to a .npz file for plotting")
    p.add_argument("--seed",  type=int, default=None, help="Random seed")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # ── load inputs ───────────────────────────────────────────────────────────
    print(f"Loading frames  : {args.frames}")
    frames = load_frames(args.frames)
    print(f"  {len(frames)} frames")

    print(f"Loading GT ({args.gt_format}): {args.gt}")
    if args.gt_format == "kitti":
        gt_data = load_gt_kitti(args.gt)
    else:
        gt_data = load_gt_tum(args.gt)
    print(f"  {len(gt_data)} GT entries")

    solver = Solver(args.config_path, config_type=args.camera)

    start, end = args.start, args.end
    if start < 0 or end >= len(frames) or start >= end:
        print(f"Error: invalid range [{start}, {end}] for {len(frames)} frames")
        sys.exit(1)

    frame_range = range(start, end + 1)

    # ── TUM: build timestamp → GT index map ───────────────────────────────────
    if args.gt_format == "tum":
        tum_map = build_tum_frame_map(gt_data, frames, frame_range, args.ts_scale)

    def get_gt_relative(i, j):
        if args.gt_format == "kitti":
            if i >= len(gt_data) or j >= len(gt_data):
                return None, None
            return kitti_relative_pose(gt_data, i, j)
        else:
            gi = tum_map.get(i)
            gj = tum_map.get(j)
            if gi is None or gj is None:
                return None, None
            return tum_relative_pose(gt_data, gi, gj)

    # ── per-pair evaluation ───────────────────────────────────────────────────
    # Each entry in pair_results is a dict with keys:
    #   frame_i, R_best, t_best,
    #   best_{combined,rot,t}, median_{combined,rot,t}, worst_{combined,rot,t},
    #   base_{combined,rot,t}, n_matches, n_trials_ok
    pair_results = []
    n_skipped = 0

    for i in tqdm(range(start, end), desc="Evaluating pairs"):
        j = i + 1
        f_prev = frames[i]
        f_curr = frames[j]

        if f_curr.matches is None or f_prev.kpts is None or f_curr.kpts is None:
            n_skipped += 1
            continue

        kpts0 = f_prev.kpts
        kpts1 = f_curr.kpts
        if hasattr(kpts0, 'numpy'):
            kpts0 = kpts0.numpy()
        if hasattr(kpts1, 'numpy'):
            kpts1 = kpts1.numpy()
        kpts0 = np.asarray(kpts0)
        kpts1 = np.asarray(kpts1)

        matches = f_curr.matches
        if hasattr(matches, 'numpy'):
            matches = matches.numpy()
        matches = np.asarray(matches)

        n_matches = len(matches)
        n_sample = min(args.N, n_matches)

        if n_sample < 5:
            tqdm.write(f"  [SKIP] pair ({i},{j}): {n_matches} matches < 5")
            n_skipped += 1
            continue

        R_gt, t_gt = get_gt_relative(i, j)
        if R_gt is None:
            tqdm.write(f"  [SKIP] pair ({i},{j}): GT not available")
            n_skipped += 1
            continue
        if np.linalg.norm(t_gt) < 1e-9:
            tqdm.write(f"  [SKIP] pair ({i},{j}): zero GT translation")
            n_skipped += 1
            continue

        all_m0 = kpts0[matches[:, 0]]
        all_m1 = kpts1[matches[:, 1]]

        # ── baseline: all matches ─────────────────────────────────────────────
        base_combined = base_rot = base_t = float('nan')
        try:
            pose_base, _ = solver.solve_relative_pose(all_m0, all_m1)
            base_rot = rotation_error_deg(pose_base.R, R_gt)
            base_t   = translation_error_deg(pose_base.t, t_gt)
            base_combined = base_rot + base_t
        except Exception:
            pass

        # ── K random trials ───────────────────────────────────────────────────
        # NaN-pad failed trials so arrays are always length K — makes the
        # per-pair error distribution exportable as a uniform [n_pairs, K] array.
        nan = float('nan')
        tc = np.full(args.K, nan)
        tr = np.full(args.K, nan)
        tt = np.full(args.K, nan)
        best_combined    = np.inf
        best_rot_err     = np.inf
        best_t_err       = np.inf
        best_R = best_t_vec = best_sub_matches = None

        for k in range(args.K):
            idx = np.random.choice(n_matches, size=n_sample, replace=False)
            sub = matches[idx]
            m0 = kpts0[sub[:, 0]]
            m1 = kpts1[sub[:, 1]]

            try:
                pose, _ = solver.solve_relative_pose(m0, m1)
                R_est, t_est = pose.R, pose.t
            except Exception:
                continue   # leave tc[k] as NaN

            r_err = rotation_error_deg(R_est, R_gt)
            t_err = translation_error_deg(t_est, t_gt)
            combined = r_err + t_err

            tc[k] = combined
            tr[k] = r_err
            tt[k] = t_err

            if combined < best_combined:
                best_combined    = combined
                best_rot_err     = r_err
                best_t_err       = t_err
                best_R           = R_est.copy()
                best_t_vec       = t_est.copy()
                best_sub_matches = sub.copy()

        if best_R is None:
            tqdm.write(f"  [SKIP] pair ({i},{j}): all {args.K} trials failed")
            n_skipped += 1
            continue

        pair_results.append(dict(
            frame_i=i,
            R_best=best_R, t_best=best_t_vec,
            best_sub_matches=best_sub_matches,
            best_combined=best_combined,
            best_rot=best_rot_err,
            best_t=best_t_err,
            median_combined=float(np.nanmedian(tc)),
            median_rot=float(np.nanmedian(tr)),
            median_t=float(np.nanmedian(tt)),
            worst_combined=float(np.nanmax(tc)),
            worst_rot=float(np.nanmax(tr)),
            worst_t=float(np.nanmax(tt)),
            base_combined=base_combined,
            base_rot=base_rot,
            base_t=base_t,
            n_matches=n_matches,
            n_trials_ok=int(np.sum(np.isfinite(tc))),
            trial_combined=tc,
            trial_rot=tr,
            trial_t=tt,
        ))

    print(f"\n{len(pair_results)} valid pairs evaluated, {n_skipped} skipped")

    if not pair_results:
        print("No valid pairs — exiting.")
        sys.exit(1)

    # ── aggregate stats helpers ───────────────────────────────────────────────
    def _stats(values):
        v = np.asarray([x for x in values if np.isfinite(x)])
        if len(v) == 0:
            return "  n/a"
        return (f"  mean={np.mean(v):.2f}°  median={np.median(v):.2f}°"
                f"  std={np.std(v):.2f}°  min={np.min(v):.2f}°  max={np.max(v):.2f}°")

    def _print_stats_block(label, results, key_comb, key_rot, key_t):
        print(f"  {label}:")
        print(f"    combined  {_stats([r[key_comb] for r in results])}")
        print(f"    rotation  {_stats([r[key_rot]  for r in results])}")
        print(f"    trans     {_stats([r[key_t]    for r in results])}")

    # ── overall summary (all valid pairs) ─────────────────────────────────────
    print()
    print("=" * 70)
    print(f"Overall stats — all {len(pair_results)} valid pairs")
    print("=" * 70)
    _print_stats_block(
        f"Subsampled N={args.N} (best of K={args.K} trials)",
        pair_results, "best_combined", "best_rot", "best_t",
    )
    _print_stats_block(
        f"Subsampled N={args.N} (median of K={args.K} trials)",
        pair_results, "median_combined", "median_rot", "median_t",
    )
    _print_stats_block(
        "Baseline (all matched keypoints)",
        pair_results, "base_combined", "base_rot", "base_t",
    )

    # ── select L best by combined error ───────────────────────────────────────
    pair_results.sort(key=lambda r: r["best_combined"])
    L = min(args.L, len(pair_results))
    selected = pair_results[:L]

    print()
    print("=" * 70)
    print(f"Selected L={L} best pairs  (all errors in degrees)")
    print("=" * 70)

    col = "{:>10}"
    hdr = (f"  {'pair':>10}  "
           f"{'sub-best':>9}  {'sub-med':>8}  {'sub-worst':>9}  "
           f"{'base':>9}  {'Δ(base-best)':>12}  "
           f"{'N_matches':>9}  {'K_ok':>5}")
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)
    for r in selected:
        i = r["frame_i"]
        delta = (r["base_combined"] - r["best_combined"]
                 if np.isfinite(r["base_combined"]) else float("nan"))
        print(
            f"  ({i:4d},{i+1:4d})  "
            f"{r['best_combined']:9.3f}°  {r['median_combined']:8.3f}°  {r['worst_combined']:9.3f}°  "
            f"{r['base_combined']:9.3f}°  {delta:+12.3f}°  "
            f"{r['n_matches']:9d}  {r['n_trials_ok']:5d}"
        )

    print()
    print(f"Selected {L} pairs — summary")
    _print_stats_block(
        f"Subsampled N={args.N} (best of K={args.K})",
        selected, "best_combined", "best_rot", "best_t",
    )
    _print_stats_block(
        "Baseline (all matched keypoints)",
        selected, "base_combined", "base_rot", "base_t",
    )

    # ── assemble output frames ────────────────────────────────────────────────
    # Each selected pair emits exactly 2 consecutive frames with kpts trimmed to
    # the N subsampled points used in the best trial, so frametool can visualize
    # them directly.  No deduplication: a frame that borders two selected pairs
    # will appear twice, once per pair context.
    selected_sorted = sorted(selected, key=lambda r: r["frame_i"])

    out_frames = []
    for r in selected_sorted:
        i        = r["frame_i"]
        j        = i + 1
        R_best   = r["R_best"]
        t_best   = r["t_best"]
        sub      = r["best_sub_matches"]   # shape [N_sub, 2], numpy int array
        N_sub    = len(sub)

        # prev frame — keep only the N_sub keypoints that were used.
        # Clear matches so frametool raises cleanly if called on this index
        # instead of silently using stale full-set indices.
        f_prev = copy.copy(frames[i])
        orig_kpts_i = frames[i].kpts
        if isinstance(orig_kpts_i, torch.Tensor):
            f_prev.kpts = orig_kpts_i[torch.from_numpy(sub[:, 0]).long()]
        else:
            f_prev.kpts = np.asarray(orig_kpts_i)[sub[:, 0]]
        f_prev.features = None
        f_prev.matches = None

        # curr frame — same, plus identity matches and best-trial pose
        f_curr = copy.copy(frames[j])
        orig_kpts_j = frames[j].kpts
        if isinstance(orig_kpts_j, torch.Tensor):
            f_curr.kpts = orig_kpts_j[torch.from_numpy(sub[:, 1]).long()]
        else:
            f_curr.kpts = np.asarray(orig_kpts_j)[sub[:, 1]]
        f_curr.features = None

        # identity matches: kpt k in prev ↔ kpt k in curr
        identity = np.stack([np.arange(N_sub), np.arange(N_sub)], axis=1)
        orig_matches = frames[j].matches
        if isinstance(orig_matches, torch.Tensor):
            f_curr.matches = torch.tensor(identity, dtype=orig_matches.dtype)
        else:
            f_curr.matches = identity

        n = np.linalg.norm(t_best)
        f_curr.E = (R_best, t_best / n if n > 1e-9 else t_best)

        out_frames.extend([f_prev, f_curr])

    if args.errors_out is not None:
        M = len(pair_results)
        K = args.K
        pair_indices   = np.array([r["frame_i"]      for r in pair_results], dtype=np.int32)
        all_combined   = np.stack([r["trial_combined"] for r in pair_results])   # [M, K]
        all_rot        = np.stack([r["trial_rot"]      for r in pair_results])
        all_t          = np.stack([r["trial_t"]        for r in pair_results])
        base_combined  = np.array([r["base_combined"]  for r in pair_results])
        base_rot       = np.array([r["base_rot"]       for r in pair_results])
        base_t         = np.array([r["base_t"]         for r in pair_results])
        np.savez(
            args.errors_out,
            pair_indices=pair_indices,
            trial_combined=all_combined,
            trial_rot=all_rot,
            trial_t=all_t,
            base_combined=base_combined,
            base_rot=base_rot,
            base_t=base_t,
            N=np.int32(args.N),
            K=np.int32(K),
            L=np.int32(args.L),
        )
        print(f"Error arrays saved → {args.errors_out}  (shape {M}×{K})")

    print(f"\nSaving {len(out_frames)} frames → {args.output}")
    export_frames(out_frames, args.output)

    print()
    print("frametool index map")
    print("  'show matches' / 'show keypoints' → use the PREV index (even)")
    print("  'info pose'                        → use the CURR index (odd)")
    print(f"  {'prev':>6}  {'curr':>6}  {'original pair':>14}  {'combined':>10}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*14}  {'-'*10}")
    for k, r in enumerate(selected_sorted):
        i = r["frame_i"]
        print(f"  {2*k:>6}  {2*k+1:>6}  ({i:4d}, {i+1:4d})      {r['best_combined']:9.3f}°")
    print("Done.")


if __name__ == "__main__":
    main()
