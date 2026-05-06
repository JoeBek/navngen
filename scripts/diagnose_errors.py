#!/usr/bin/env python3
"""
Find the top-K N-frame windows with highest RTE/RRE, then save keypoint
and match visualizations for the frames in those windows.

Outputs per window:
  frame_XXXXX_kpts.png          — image with keypoints (green=matched, blue=unmatched)
  frame_XXXXX_to_YYYYY_matches.png — side-by-side match visualization between consecutive frames
  error_summary.json            — all window errors and frame indices
"""

import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.navngen.export_trajectory import load_frames


# ── trajectory helpers ────────────────────────────────────────────────────────

def load_tum(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def tum_to_poses(tum: np.ndarray):
    """Returns list of (ts, R (3x3), t (3,)) from an Nx8 TUM array."""
    poses = []
    for row in tum:
        ts = row[0]
        t = row[1:4]
        q = row[4:8]  # qx qy qz qw
        R = Rotation.from_quat(q).as_matrix()
        poses.append((ts, R, t))
    return poses


def match_timestamps(gt_poses, est_poses, max_dt=None):
    """
    For each GT pose find the nearest EST pose by timestamp.
    Returns list of (gt_idx, est_idx) pairs, optionally filtered by max_dt.
    """
    est_ts = np.array([p[0] for p in est_poses])
    pairs = []
    for gi, (gt_ts, _, _) in enumerate(gt_poses):
        ei = int(np.argmin(np.abs(est_ts - gt_ts)))
        if max_dt is None or abs(est_ts[ei] - gt_ts) <= max_dt:
            pairs.append((gi, ei))
    return pairs


def relative_pose(R_i, t_i, R_j, t_j):
    """T_ij = T_wi^{-1} @ T_wj  (pose of j expressed in frame of i)."""
    R_rel = R_i.T @ R_j
    t_rel = R_i.T @ (t_j - t_i)
    return R_rel, t_rel


def rotation_error_deg(R_err: np.ndarray) -> float:
    val = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))


# ── error computation ─────────────────────────────────────────────────────────

def compute_window_errors(gt_poses, est_poses, pairs, N: int):
    """
    For each window of N consecutive matched pairs compute RTE and RRE.

    T_err = T_gt_rel^{-1} @ T_est_rel
    RTE = ||t_err||,  RRE = rotation angle of R_err (degrees)
    """
    results = []
    for i in range(len(pairs) - N + 1):
        window = pairs[i:i + N]
        gi0, ei0 = window[0]
        gi1, ei1 = window[-1]

        _, R_gt0, t_gt0   = gt_poses[gi0]
        _, R_gt1, t_gt1   = gt_poses[gi1]
        _, R_est0, t_est0 = est_poses[ei0]
        _, R_est1, t_est1 = est_poses[ei1]

        R_gt_rel,  t_gt_rel  = relative_pose(R_gt0,  t_gt0,  R_gt1,  t_gt1)
        R_est_rel, t_est_rel = relative_pose(R_est0, t_est0, R_est1, t_est1)

        R_err = R_gt_rel.T @ R_est_rel
        t_err = R_gt_rel.T @ (t_est_rel - t_gt_rel)

        results.append({
            'start_pair_idx': i,
            'gt_indices':  [p[0] for p in window],
            'est_indices': [p[1] for p in window],
            'rte': float(np.linalg.norm(t_err)),
            'rre': rotation_error_deg(R_err),
        })
    return results


# ── visualization helpers ──────────────────────────────────────────────────────

def draw_kpts_overlay(image_path: Path, kpts: np.ndarray,
                      match_flags: np.ndarray) -> np.ndarray | None:
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    r = 5
    for i, (x, y) in enumerate(kpts):
        color = (0, 255, 0) if match_flags[i] else (255, 0, 0)
        xi, yi = int(x), int(y)
        cv2.rectangle(img, (xi - r, yi - r), (xi + r, yi + r), color, 1)
        cv2.circle(img, (xi, yi), 2, color, -1)
    return img


def draw_matches_pair(img0_path: Path, img1_path: Path,
                      kpts0: np.ndarray, kpts1: np.ndarray,
                      matches: np.ndarray) -> np.ndarray | None:
    img0 = cv2.imread(str(img0_path))
    img1 = cv2.imread(str(img1_path))
    if img0 is None or img1 is None:
        return None

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    h = max(h0, h1)
    canvas = np.zeros((h, w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0]      = img0
    canvas[:h1, w0:w0+w1] = img1

    for m in matches:
        x0, y0 = int(kpts0[m[0], 0]), int(kpts0[m[0], 1])
        x1, y1 = int(kpts1[m[1], 0]) + w0, int(kpts1[m[1], 1])
        cv2.line(canvas,   (x0, y0), (x1, y1), (0, 200, 200), 1, cv2.LINE_AA)
        cv2.circle(canvas, (x0, y0), 3, (0, 255, 0), -1)
        cv2.circle(canvas, (x1, y1), 3, (0, 255, 0), -1)
    return canvas


def annotate(img: np.ndarray, text: str) -> None:
    """Draw text with a dark outline for readability."""
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_PLAIN, 1.3,
                (0,   0,   0  ), 2, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_PLAIN, 1.3,
                (255, 255, 255), 1, cv2.LINE_AA)


def to_np(t) -> np.ndarray:
    return t.numpy() if hasattr(t, 'numpy') else np.asarray(t)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Find top-K high-error N-frame windows and save diagnostic images.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--gt",         type=Path, required=True,
                        help="Ground truth TUM trajectory file.")
    parser.add_argument("--est",        type=Path, required=True,
                        help="Estimated TUM trajectory file.")
    parser.add_argument("--frames",     type=Path, default=None,
                        help="Pickled frame sequence (.pkl.gz). Required unless --dry_run.")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Directory to write outputs. Required unless --dry_run.")
    parser.add_argument("--N",          type=int,   default=5,
                        help="Window size in frames (default 5).")
    parser.add_argument("--K",          type=int,   default=10,
                        help="Number of top-error windows to save (default 10).")
    parser.add_argument("--sort_by",    choices=['rte', 'rre'], default='rte',
                        help="Metric used to rank windows (default rte).")
    parser.add_argument("--ts_scale",   type=float, default=1.0,
                        help="Scale applied to EST timestamps before matching GT.\n"
                             "Use 1e9 when GT is in nanoseconds and EST is in seconds.")
    parser.add_argument("--max_dt",     type=float, default=None,
                        help="Max allowed timestamp difference for a match (GT units).")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Print window errors and frame indices without loading frames or saving anything.")
    args = parser.parse_args()

    if not args.dry_run:
        if args.frames is None or args.output_dir is None:
            parser.error("--frames and --output_dir are required unless --dry_run is set.")
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── load trajectories ──
    gt_tum  = load_tum(args.gt)
    est_tum = load_tum(args.est)

    gt_poses  = tum_to_poses(gt_tum)
    est_tum_s = est_tum.copy()
    est_tum_s[:, 0] *= args.ts_scale          # scale EST timestamps to GT units
    est_poses = tum_to_poses(est_tum_s)

    pairs = match_timestamps(gt_poses, est_poses, max_dt=args.max_dt)
    print(f"Matched {len(pairs)} pose pairs  (GT={len(gt_poses)}, EST={len(est_poses)})")

    if len(pairs) < args.N:
        print(f"Not enough matched pairs ({len(pairs)}) for N={args.N}. Exiting.")
        sys.exit(1)

    # ── compute errors ──
    window_errors = compute_window_errors(gt_poses, est_poses, pairs, args.N)
    window_errors.sort(key=lambda w: w[args.sort_by], reverse=True)
    top_k = window_errors[:args.K]

    print(f"\nTop {args.K} windows by {args.sort_by.upper()} (N={args.N}):")
    for i, w in enumerate(top_k):
        gi0 = w['gt_indices'][0]
        print(f"  [{i+1:2d}] pair={w['start_pair_idx']:4d}  "
              f"GT_ts={gt_poses[gi0][0]:.3f}  "
              f"RTE={w['rte']:.4f}  RRE={w['rre']:.2f}°")

    if args.dry_run:
        return

    # ── load frames ──
    print(f"\nLoading frames from {args.frames} …")
    frames = load_frames(args.frames)
    print(f"Loaded {len(frames)} frames.")

    # Map EST traj index → frame index (match by timestamp in seconds)
    # EST TUM timestamps are in the original units of the TUM file (seconds for campus).
    # frame.timestamp / 1e9 converts to the same units.
    frame_ts = np.array([f.timestamp / 1e9 if f.timestamp is not None else float('nan')
                         for f in frames])

    def est_to_frame(ei: int) -> int:
        return int(np.argmin(np.abs(frame_ts - est_tum[ei, 0])))

    # ── build summary ──
    summary = []
    for rank, w in enumerate(top_k):
        fis = [est_to_frame(ei) for ei in w['est_indices']]
        print(f"  rank {rank+1:2d}: est_indices={w['est_indices']}  frame_indices={fis}")
        summary.append({
            'rank':            rank + 1,
            'start_pair_idx':  w['start_pair_idx'],
            'rte':             w['rte'],
            'rre':             w['rre'],
            'gt_ts_start':     float(gt_poses[w['gt_indices'][0]][0]),
            'gt_ts_end':       float(gt_poses[w['gt_indices'][-1]][0]),
            'frame_indices':   fis,
        })

    summary_path = args.output_dir / 'error_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {summary_path}")

    # ── save visualizations ──
    kpts_saved = set()   # track which per-frame kpt images have been written

    for rank, (w, entry) in enumerate(zip(top_k, summary)):
        window_dir = args.output_dir / f"rank_{rank+1:02d}_rte{w['rte']:.3f}_rre{w['rre']:.1f}"
        window_dir.mkdir(exist_ok=True)

        fis = entry['frame_indices']

        for pos, fi in enumerate(fis):
            if fi >= len(frames):
                continue
            frame = frames[fi]
            if frame.path is None or not Path(frame.path).exists():
                continue

            # ── per-frame keypoint overlay ──
            if fi not in kpts_saved and frame.kpts is not None:
                kpts_np = to_np(frame.kpts)
                match_flags = np.zeros(len(kpts_np), dtype=bool)
                if frame.matches is not None:
                    m = to_np(frame.matches)
                    if m.ndim == 2 and m.shape[1] >= 2:
                        idx1 = m[:, 1].astype(int)
                        idx1 = idx1[idx1 < len(kpts_np)]
                        match_flags[idx1] = True

                img = draw_kpts_overlay(frame.path, kpts_np, match_flags)
                if img is not None:
                    n_m = int(match_flags.sum())
                    annotate(img, f"frame {fi}  matched={n_m}/{len(kpts_np)}")
                    cv2.imwrite(str(window_dir / f"frame_{fi:05d}_kpts.png"), img)
                kpts_saved.add(fi)

            # ── side-by-side match visualization with previous frame ──
            if pos == 0:
                continue
            fi_prev = fis[pos - 1]
            if fi_prev >= len(frames):
                continue
            frame_prev = frames[fi_prev]

            if (frame_prev.path is None or not Path(frame_prev.path).exists()
                    or frame.matches is None
                    or frame_prev.kpts is None
                    or frame.kpts is None):
                continue

            kpts0 = to_np(frame_prev.kpts)
            kpts1 = to_np(frame.kpts)
            m = to_np(frame.matches)

            if m.ndim == 2 and m.shape[1] >= 2:
                pair_img = draw_matches_pair(frame_prev.path, frame.path, kpts0, kpts1, m)
                if pair_img is not None:
                    annotate(pair_img, f"frames {fi_prev}→{fi}  matches={len(m)}")
                    out = window_dir / f"frame_{fi_prev:05d}_to_{fi:05d}_matches.png"
                    cv2.imwrite(str(out), pair_img)

    print(f"Visualizations → {args.output_dir}")


if __name__ == "__main__":
    main()
