#!/usr/bin/env python3
"""
plot_keypoint_heatmap.py

For a given frame pair (I, I+1), run K random subsamples of N keypoints and
visualise how often each image region is sampled as a heatmap overlaid on the
image pair.

Usage:
    python3 scripts/plot_keypoint_heatmap.py <frames.pkl.gz> \\
        --frame I -N N -K K \\
        [--sigma FLOAT]   smoothing sigma in pixels (default: 15)
        [--alpha FLOAT]   heatmap opacity, 0-1 (default: 0.55)
        [--seed INT]
        [--save PATH]     save figure instead of displaying
        [--cmap NAME]     matplotlib colormap (default: inferno)
        [--show_kpts]     also scatter all candidate keypoints in grey

    To restrict the heatmap to only the best-scoring trials by pose error,
    add --top T alongside the GT / solver flags:
        --top T           keep only the T lowest-error trials
        --gt PATH         ground-truth pose file
        --gt_format       kitti|tum (default: tum)
        --ts_scale FLOAT  timestamp scale factor (default: 1.0)
        --config_path     camera calibration YAML
        --camera          surfnav|kitti|euroc (default: surfnav)
        --metric          translation|rotation|combined (default: combined)
"""

import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import matplotlib.cm as cm

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.navngen.export_trajectory import load_frames
from src.navngen.load_images import load_image


# ─────────────────────────── helpers ─────────────────────────────────────────

def to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def build_heatmap(coords_xy, h, w, sigma):
    """
    coords_xy : (M, 2) float array of (x, y) pixel coords
    Returns a (h, w) float array in [0, 1].
    """
    x = np.clip(np.round(coords_xy[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(coords_xy[:, 1]).astype(int), 0, h - 1)

    grid = np.zeros((h, w), dtype=np.float32)
    np.add.at(grid, (y, x), 1.0)

    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma)

    if grid.max() > 0:
        grid /= grid.max()
    return grid


def image_tensor_to_rgb(img_t):
    """Convert (C, H, W) torch tensor in [0,1] to (H, W, 3) uint8."""
    arr = to_numpy(img_t)
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)  # CHW → HWC
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 1:
        arr = np.concatenate([arr] * 3, axis=-1)
    return arr


def overlay_heatmap(ax, rgb, heatmap, alpha, cmap_name, title, kpts_all=None):
    ax.imshow(rgb)
    heat_rgba = cm.get_cmap(cmap_name)(heatmap)
    heat_rgba[..., 3] = heatmap * alpha          # transparent where cold
    ax.imshow(heat_rgba)
    if kpts_all is not None:
        ax.scatter(kpts_all[:, 0], kpts_all[:, 1],
                   s=4, c="white", alpha=0.25, linewidths=0)
    ax.set_title(title)
    ax.axis("off")


# ─────────────────────────── GT helpers (mirrors subsample.py) ───────────────

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


def kitti_relative_pose(poses, i, j):
    rel = np.linalg.inv(poses[j]) @ poses[i]
    return rel[:3, :3], rel[:3, 3]


def tum_relative_pose(entries, gi, gj):
    _, R0, t0 = entries[gi]
    _, R1, t1 = entries[gj]
    return R0.T @ R1, R0.T @ (t1 - t0)


def compute_error(R_est, t_est, R_gt, t_gt, metric):
    if metric == "translation":
        ne, ng = np.linalg.norm(t_est), np.linalg.norm(t_gt)
        if ne < 1e-9 or ng < 1e-9:
            return float("nan")
        cos_val = np.clip(np.dot(t_est / ne, t_gt / ng), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_val)))
    elif metric == "rotation":
        dR = R_est.T @ R_gt
        cos_val = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_val)))
    else:  # combined
        ne, ng = np.linalg.norm(t_est), np.linalg.norm(t_gt)
        if ne < 1e-9 or ng < 1e-9:
            t_err = float("nan")
        else:
            cos_val = np.clip(np.dot(t_est / ne, t_gt / ng), -1.0, 1.0)
            t_err = float(np.degrees(np.arccos(cos_val)))
        dR = R_est.T @ R_gt
        cos_val = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
        r_err = float(np.degrees(np.arccos(cos_val)))
        return t_err + r_err


# ─────────────────────────── main ────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(prog="plot_keypoint_heatmap")
    p.add_argument("frames",   type=Path, help="Input .pkl.gz frame file")
    p.add_argument("--frame",  type=int, required=True, dest="frame_i",
                   help="Index of the first frame in the pair (second is frame+1)")
    p.add_argument("-N",       type=int, required=True, dest="N",
                   help="Keypoints to subsample per trial")
    p.add_argument("-K",       type=int, required=True, dest="K",
                   help="Number of random trials")
    p.add_argument("--sigma",  type=float, default=15.0,
                   help="Gaussian smoothing sigma in pixels (default: 15)")
    p.add_argument("--alpha",  type=float, default=0.55,
                   help="Max heatmap opacity, 0–1 (default: 0.55)")
    p.add_argument("--cmap",   default="inferno",
                   help="Matplotlib colormap (default: inferno)")
    p.add_argument("--seed",   type=int, default=None)
    p.add_argument("--save",   type=Path, default=None,
                   help="Save figure to PATH instead of displaying")
    p.add_argument("--show_kpts", action="store_true",
                   help="Scatter all candidate keypoints in white behind the heatmap")
    # ── best-N filtering (requires GT + solver) ───────────────────────────────
    p.add_argument("--top",         type=int, default=None, metavar="T",
                   help="Keep only the T lowest-error trials for the heatmap "
                        "(requires --gt and --config_path)")
    p.add_argument("--gt",          type=Path, default=None,
                   help="Ground-truth pose file (required with --top)")
    p.add_argument("--gt_format",   default="tum", choices=["kitti", "tum"])
    p.add_argument("--ts_scale",    type=float, default=1.0,
                   help="Multiply frame.timestamp by this to get seconds "
                        "(use 1e-9 for nanosecond timestamps; TUM only)")
    p.add_argument("--config_path", type=Path, default=None,
                   help="Camera calibration YAML (required with --top)")
    p.add_argument("--camera",      default="surfnav",
                   choices=["surfnav", "kitti", "euroc"])
    p.add_argument("--metric",      default="combined",
                   choices=["translation", "rotation", "combined"],
                   help="Error metric used to rank trials (default: combined)")
    return p


def main():
    args = build_parser().parse_args()

    if args.top is not None:
        if args.gt is None or args.config_path is None:
            print("Error: --top requires both --gt and --config_path")
            sys.exit(1)

    if args.seed is not None:
        np.random.seed(args.seed)

    print(f"Loading frames: {args.frames}")
    frames = load_frames(args.frames)
    print(f"  {len(frames)} frames total")

    i, j = args.frame_i, args.frame_i + 1
    if i < 0 or j >= len(frames):
        print(f"Error: frame index {i} out of range for {len(frames)} frames")
        sys.exit(1)

    f_prev, f_curr = frames[i], frames[j]

    if f_prev.kpts is None or f_curr.kpts is None or f_curr.matches is None:
        print("Error: missing keypoints or matches on the requested frame pair")
        sys.exit(1)

    kpts0   = to_numpy(f_prev.kpts)    # (P, 2) — all kpts in frame i
    kpts1   = to_numpy(f_curr.kpts)    # (Q, 2) — all kpts in frame j
    matches = to_numpy(f_curr.matches)  # (M, 2) — index pairs into kpts0/kpts1

    n_matches = len(matches)
    n_sample  = min(args.N, n_matches)

    if n_sample < 5:
        print(f"Error: only {n_matches} matches — need at least 5")
        sys.exit(1)

    print(f"Pair ({i}, {j}): {n_matches} matches, subsampling N={n_sample} over K={args.K} trials")

    # ── optionally load GT + solver for --top filtering ───────────────────────
    solver = None
    R_gt = t_gt = None
    if args.top is not None:
        from src.navngen.trajectory import Solver

        print(f"Loading GT ({args.gt_format}): {args.gt}")
        gt_data = (load_gt_kitti(args.gt) if args.gt_format == "kitti"
                   else load_gt_tum(args.gt))
        print(f"  {len(gt_data)} GT entries")

        solver = Solver(args.config_path, config_type=args.camera)

        if args.gt_format == "kitti":
            if i < len(gt_data) and j < len(gt_data):
                R_gt, t_gt = kitti_relative_pose(gt_data, i, j)
        else:
            gt_ts = np.array([e[0] for e in gt_data])
            def nearest(frame_idx):
                ts = getattr(frames[frame_idx], "timestamp", None)
                if ts is None:
                    return None
                return int(np.argmin(np.abs(gt_ts - float(ts) * args.ts_scale)))
            gi, gj = nearest(i), nearest(j)
            if gi is not None and gj is not None:
                R_gt, t_gt = tum_relative_pose(gt_data, gi, gj)

        if R_gt is None or np.linalg.norm(t_gt) < 1e-9:
            print("Error: could not obtain valid GT relative pose for this pair")
            sys.exit(1)

    # ── run K trials ──────────────────────────────────────────────────────────
    # Each trial stores its keypoint coords; if --top is set, also stores error.
    trials = []   # list of (err_or_None, coords0, coords1)

    for _ in range(args.K):
        idx = np.random.choice(n_matches, size=n_sample, replace=False)
        sub = matches[idx]
        c0  = kpts0[sub[:, 0]]
        c1  = kpts1[sub[:, 1]]

        err = None
        if solver is not None:
            try:
                pose, _ = solver.solve_relative_pose(c0, c1)
                err = compute_error(pose.R, pose.t, R_gt, t_gt, args.metric)
            except Exception:
                err = float("nan")

        trials.append((err, c0, c1))

    # ── filter to top-T trials if requested ───────────────────────────────────
    if args.top is not None:
        valid = [(e, c0, c1) for e, c0, c1 in trials if np.isfinite(e)]
        if not valid:
            print("Error: all trials failed pose estimation — cannot rank by error")
            sys.exit(1)
        valid.sort(key=lambda t: t[0])
        top_t = min(args.top, len(valid))
        if top_t < args.top:
            print(f"Warning: only {top_t} trials succeeded (requested --top {args.top})")
        trials = valid[:top_t]
        best_err  = trials[0][0]
        worst_err = trials[-1][0]
        print(f"Keeping top {top_t} trials by {args.metric} error  "
              f"(best={best_err:.3f}°  worst={worst_err:.3f}°)")

    selected0 = np.concatenate([c0 for _, c0, _ in trials], axis=0)

    # Load image
    img0 = image_tensor_to_rgb(load_image(f_prev.path))
    h0, w0 = img0.shape[:2]

    hmap0 = build_heatmap(selected0, h0, w0, args.sigma)

    # ── plot ──────────────────────────────────────────────────────────────────
    n_used = len(trials)
    if args.top is not None:
        filter_str = f"top {n_used} of {args.K} trials by {args.metric} error"
    else:
        filter_str = f"{args.K} trials (all)"

    fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle(
        f"Keypoint subsample heatmap  —  N={args.N}, {filter_str}\n"
        f"frame {i}  |  {n_matches} total matches",
        fontsize=11,
    )

    kpts_all0 = kpts0[matches[:, 0]] if args.show_kpts else None

    overlay_heatmap(ax0, img0, hmap0, args.alpha, args.cmap,
                    f"Frame {i}", kpts_all0)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=args.cmap,
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax0, fraction=0.025, pad=0.02,
                 label="relative selection frequency")

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
