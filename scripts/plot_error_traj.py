#!/usr/bin/env python3
"""
plot_error_traj.py

Two modes:

  Error mode (default): plot the GT trajectory colored by the per-pose error
  of an estimated trajectory.  The estimated path is NOT drawn — error is
  encoded as a color gradient (green = low, red = high).

  Rotation mode (--show_rot gt|est): plot a single trajectory as a plain line
  with heading arrows showing camera orientation at every N-th pose.  No error
  is computed; when est is selected the trajectory is Sim(3)-aligned to GT
  first (requires both files).

Error is computed after optional Sim(3) alignment (on by default; --no_align
to disable).

  translation:  Euclidean distance (m) between aligned estimated and GT
                position at each matched pose.
  rotation:     Geodesic angle (°) between aligned estimated and GT
                orientation at each matched pose.

Trajectory formats:
  tum    — "timestamp tx ty tz qx qy qz qw"
  kitti  — one "r11 r12 ... tz" (12-value) row per frame (camera-to-world 3×4)

Usage:
    # error mode
    python3 scripts/plot_error_traj.py <gt> <est> \\
        [--gt_format tum|kitti]  [--est_format tum|kitti] \\
        [--metric translation|rotation] \\
        [--plane xz|xy|yz]  [--no_align] \\
        [--vmax FLOAT]  [--cmap NAME] \\
        [--ts_scale FLOAT]  [--label STR] \\
        [--title STR]  [--save PATH]

    # rotation mode — GT only (est not required)
    python3 scripts/plot_error_traj.py <gt> --show_rot gt \\
        [--rot_every N]  [--rot_scale FLOAT]  [--plane xz]  [--save PATH]

    # rotation mode — estimated trajectory (aligned to GT)
    python3 scripts/plot_error_traj.py <gt> <est> --show_rot est \\
        [--rot_every N]  [--rot_scale FLOAT]  [--plane xz]  [--save PATH]
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from scipy.spatial.transform import Rotation

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


# ─────────────────────────── trajectory loading ──────────────────────────────

def load_tum(path: Path):
    """Return list of (timestamp_s, R 3×3, t 3) — camera-to-world."""
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


def load_kitti(path: Path):
    """Return list of 4×4 numpy arrays (camera-to-world) one per frame."""
    poses = []
    with open(path) as f:
        for line in f:
            vals = [float(v) for v in line.strip().split()]
            if not vals:
                continue
            P = np.array(vals).reshape(3, 4)
            poses.append(np.vstack([P, [0, 0, 0, 1]]))
    return poses


def tum_to_arrays(entries, ts_scale=1.0):
    """(timestamps N, positions N×3, rotations N×3×3)."""
    ts  = np.array([e[0] * ts_scale for e in entries])
    pos = np.array([e[2]            for e in entries])
    rot = np.array([e[1]            for e in entries])
    return ts, pos, rot


def kitti_to_arrays(poses):
    """(indices N, positions N×3, rotations N×3×3)."""
    idx = np.arange(len(poses))
    pos = np.array([p[:3, 3] for p in poses])
    rot = np.array([p[:3, :3] for p in poses])
    return idx, pos, rot


def load_traj(path, fmt, ts_scale=1.0):
    if fmt == "tum":
        entries = load_tum(path)
        _, pos, rot = tum_to_arrays(entries, ts_scale=ts_scale)
        return entries, pos, rot
    else:
        poses = load_kitti(path)
        _, pos, rot = kitti_to_arrays(poses)
        return poses, pos, rot


# ─────────────────────────── pose matching ───────────────────────────────────

def match_tum(gt_entries, est_entries, ts_scale_gt=1.0, ts_scale_est=1.0):
    gt_ts  = np.array([e[0] * ts_scale_gt  for e in gt_entries])
    est_ts = np.array([e[0] * ts_scale_est for e in est_entries])
    gt_idx, est_idx = [], []
    for i, ts in enumerate(gt_ts):
        j = int(np.argmin(np.abs(est_ts - ts)))
        gt_idx.append(i)
        est_idx.append(j)
    return np.array(gt_idx), np.array(est_idx)


# ─────────────────────────── Sim(3) alignment (Umeyama) ──────────────────────

def umeyama_alignment(src_pos: np.ndarray, dst_pos: np.ndarray):
    """dst ≈ c * R @ src + t  →  returns (c, R, t)."""
    n = len(src_pos)
    mu_s  = src_pos.mean(axis=0)
    mu_d  = dst_pos.mean(axis=0)
    src_c = src_pos - mu_s
    dst_c = dst_pos - mu_d
    var_s = np.mean(np.sum(src_c ** 2, axis=1))
    H     = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, float(np.sign(d))])
    R = U @ D @ Vt
    c = float((S * np.diag(D)).sum() / var_s) if var_s > 1e-12 else 1.0
    t = mu_d - c * (R @ mu_s)
    return c, R, t


def apply_sim3(c, R, t, pos):
    return c * (R @ pos.T).T + t


def apply_sim3_rot(R_sim3, rotations):
    return R_sim3 @ rotations


# ─────────────────────────── error metrics ───────────────────────────────────

def translation_errors(gt_pos, est_pos_aligned):
    return np.linalg.norm(gt_pos - est_pos_aligned, axis=1)


def rotation_errors_deg(gt_rot, est_rot_aligned):
    errs = []
    for R_gt, R_est in zip(gt_rot, est_rot_aligned):
        dR = R_gt.T @ R_est
        cos_val = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
        errs.append(float(np.degrees(np.arccos(cos_val))))
    return np.array(errs)


# ─────────────────────────── plotting helpers ────────────────────────────────

PLANE_AXES   = {"xz": (0, 2), "xy": (0, 1), "yz": (1, 2)}
PLANE_LABELS = {"xz": ("x", "z"), "xy": ("x", "y"), "yz": ("y", "z")}


def colored_path(ax, positions, errors, cmap_name, vmin, vmax):
    """LineCollection path colored by per-pose error."""
    pts     = positions.reshape(-1, 1, 2)
    segs    = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seg_err = (errors[:-1] + errors[1:]) / 2.0
    norm    = plt.Normalize(vmin=vmin, vmax=vmax)
    lc      = LineCollection(segs, cmap=cmap_name, norm=norm, linewidth=2.0, zorder=2)
    lc.set_array(seg_err)
    ax.add_collection(lc)
    return lc


def draw_heading_arrows(ax, pos, rot, ax0, ax1, every, scale,
                        color="black", label=None, ref_pos=None):
    """
    Draw camera heading arrows (forward = camera +z axis projected onto plane).
    pos: (N, 3), rot: (N, 3, 3)
    ref_pos: if given, use its extent to set a common arrow length across calls.
    """
    ref       = ref_pos if ref_pos is not None else pos
    extent    = max(np.ptp(ref[:, ax0]), np.ptp(ref[:, ax1]))
    arrow_len = scale * extent
    sub       = np.arange(0, len(pos), every)
    p_sub     = pos[sub]
    r_sub     = rot[sub]
    fwd       = r_sub[:, :, 2]          # camera +z in world frame
    dx        = fwd[:, ax0]
    dy        = fwd[:, ax1]
    norms     = np.hypot(dx, dy) + 1e-9
    dx, dy    = dx / norms * arrow_len, dy / norms * arrow_len
    ax.quiver(
        p_sub[:, ax0], p_sub[:, ax1], dx, dy,
        angles="xy", scale_units="xy", scale=1,
        color=color, alpha=0.6, width=0.002,
        headwidth=5, headlength=6, zorder=4,
        label=label,
    )


# ─────────────────────────── argument parser ─────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(prog="plot_error_traj",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("gt",   type=Path, help="Ground-truth trajectory file")
    p.add_argument("est",  type=Path, nargs="?", default=None,
                   help="Estimated trajectory file (required unless --show_rot gt)")
    p.add_argument("--gt_format",  default="tum",  choices=["tum", "kitti"])
    p.add_argument("--est_format", default="tum",  choices=["tum", "kitti"])
    p.add_argument("--metric",     default="translation",
                   choices=["translation", "rotation"],
                   help="Error metric (error mode only)")
    p.add_argument("--plane",      default="xz",   choices=["xz", "xy", "yz"])
    p.add_argument("--no_align",   action="store_true",
                   help="Skip Sim(3) alignment")
    p.add_argument("--vmax",       type=float, default=None,
                   help="Max error for colormap (default: 95th-percentile)")
    p.add_argument("--cmap",       default="RdYlGn_r",
                   help="Colormap for error mode")
    p.add_argument("--ts_scale",   type=float, default=1.0,
                   help="Multiply GT timestamps by this to get seconds (TUM only)")
    p.add_argument("--label",      type=str, default=None)
    p.add_argument("--title",      type=str, default=None)
    p.add_argument("--save",       type=Path, default=None)
    # rotation visualisation
    p.add_argument("--show_rot",   choices=["gt", "est", "both"], default=None,
                   help="Rotation mode: plot trajectory/ies with heading arrows "
                        "(no error computed). 'est'/'both' require the est argument "
                        "and align est to GT first.")
    p.add_argument("--rot_every",  type=int, default=10,
                   help="Draw a heading arrow every N poses")
    p.add_argument("--rot_scale",  type=float, default=0.03,
                   help="Arrow length as fraction of trajectory extent")
    return p


# ─────────────────────────── main ────────────────────────────────────────────

def main():
    args = build_parser().parse_args()
    ax0, ax1 = PLANE_AXES[args.plane]
    xl, yl   = PLANE_LABELS[args.plane]

    # ── validate inputs ───────────────────────────────────────────────────────
    if args.show_rot != "gt" and args.est is None:
        print("Error: est is required unless --show_rot gt")
        sys.exit(1)

    # ── load GT ───────────────────────────────────────────────────────────────
    gt_raw, gt_pos, gt_rot = load_traj(args.gt, args.gt_format, ts_scale=args.ts_scale)
    print(f"GT  : {len(gt_pos)} poses  →  {args.gt.name}")

    # ── rotation mode ─────────────────────────────────────────────────────────
    if args.show_rot is not None:
        # load & align est whenever it's needed
        est_pos_aligned = est_rot_aligned = None
        if args.show_rot in ("est", "both"):
            est_raw, est_pos, est_rot = load_traj(args.est, args.est_format)
            print(f"Est : {len(est_pos)} poses  →  {args.est.name}")
            if args.gt_format == "tum" and args.est_format == "tum":
                gi, ei = match_tum(gt_raw, est_raw, ts_scale_gt=args.ts_scale)
            else:
                n = min(len(gt_pos), len(est_pos))
                gi, ei = np.arange(n), np.arange(n)
            if not args.no_align:
                c, R_sim3, t_sim3 = umeyama_alignment(est_pos[ei], gt_pos[gi])
                print(f"Sim(3) alignment: c={c:.4f}")
                est_pos_aligned = apply_sim3(c, R_sim3, t_sim3, est_pos)
                est_rot_aligned = apply_sim3_rot(R_sim3, est_rot)
            else:
                est_pos_aligned = est_pos
                est_rot_aligned = est_rot

        # reference extent for consistent arrow lengths across both trajectories
        ref_pos = gt_pos if est_pos_aligned is None else np.vstack([gt_pos, est_pos_aligned])

        fig, ax = plt.subplots(figsize=(9, 7))

        if args.show_rot in ("gt", "both"):
            ax.plot(gt_pos[:, ax0], gt_pos[:, ax1],
                    color="steelblue", linewidth=1.5, zorder=1, label="GT")
            ax.scatter(*gt_pos[0,  [ax0, ax1]], s=60, color="steelblue", zorder=5, marker="o")
            ax.scatter(*gt_pos[-1, [ax0, ax1]], s=60, color="steelblue", zorder=5, marker="s")
            draw_heading_arrows(ax, gt_pos, gt_rot, ax0, ax1,
                                args.rot_every, args.rot_scale,
                                color="steelblue", label="GT heading", ref_pos=ref_pos)

        if args.show_rot in ("est", "both"):
            ax.plot(est_pos_aligned[:, ax0], est_pos_aligned[:, ax1],
                    color="tomato", linewidth=1.5, zorder=1, label="Est")
            ax.scatter(*est_pos_aligned[0,  [ax0, ax1]], s=60, color="tomato", zorder=5, marker="o")
            ax.scatter(*est_pos_aligned[-1, [ax0, ax1]], s=60, color="tomato", zorder=5, marker="s")
            draw_heading_arrows(ax, est_pos_aligned, est_rot_aligned, ax0, ax1,
                                args.rot_every, args.rot_scale,
                                color="tomato", label="Est heading", ref_pos=ref_pos)

        est_name = args.est.name if args.est else ""
        default_title = args.label or (
            f"GT: {args.gt.name}" if args.show_rot == "gt" else
            f"Est: {est_name}" if args.show_rot == "est" else
            f"GT vs Est: {est_name}"
        )
        ax.set_title(args.title or default_title, fontsize=10)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()

        if args.save:
            plt.savefig(args.save, dpi=150, bbox_inches="tight")
            print(f"Saved → {args.save}")
        else:
            plt.show()
        return

    # ── error mode ────────────────────────────────────────────────────────────
    est_raw, est_pos, est_rot = load_traj(args.est, args.est_format)
    print(f"Est : {len(est_pos)} poses  →  {args.est.name}")

    if args.gt_format == "tum" and args.est_format == "tum":
        gi, ei = match_tum(gt_raw, est_raw, ts_scale_gt=args.ts_scale)
    else:
        n = min(len(gt_pos), len(est_pos))
        gi, ei = np.arange(n), np.arange(n)

    gt_pos_m  = gt_pos[gi]
    gt_rot_m  = gt_rot[gi]
    est_pos_m = est_pos[ei]
    est_rot_m = est_rot[ei]
    print(f"Matched {len(gi)} pose pairs")

    if not args.no_align:
        c, R_sim3, t_sim3 = umeyama_alignment(est_pos_m, gt_pos_m)
        print(f"Sim(3) alignment: c={c:.4f}")
        est_pos_aligned = apply_sim3(c, R_sim3, t_sim3, est_pos_m)
        est_rot_aligned = apply_sim3_rot(R_sim3, est_rot_m)
    else:
        est_pos_aligned = est_pos_m
        est_rot_aligned = est_rot_m

    if args.metric == "translation":
        errors = translation_errors(gt_pos_m, est_pos_aligned)
        unit   = "m"
    else:
        errors = rotation_errors_deg(gt_rot_m, est_rot_aligned)
        unit   = "°"

    print(f"Error ({args.metric}):  mean={errors.mean():.4f}{unit}  "
          f"median={np.median(errors):.4f}{unit}  "
          f"max={errors.max():.4f}{unit}")

    vmin = 0.0
    vmax = args.vmax if args.vmax is not None else float(np.percentile(errors, 95))
    if vmax <= vmin:
        vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(gt_pos[:, ax0], gt_pos[:, ax1],
            color="lightgrey", linewidth=1.0, zorder=1, label="GT path")
    lc = colored_path(ax, gt_pos_m[:, [ax0, ax1]], errors,
                      cmap_name=args.cmap, vmin=vmin, vmax=vmax)
    ax.scatter(*gt_pos_m[0,  [ax0, ax1]], s=60, color="black", zorder=5,
               label="start", marker="o")
    ax.scatter(*gt_pos_m[-1, [ax0, ax1]], s=60, color="black", zorder=5,
               label="end",   marker="s")

    cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(f"translation error ({unit})" if args.metric == "translation"
                   else f"rotation error ({unit})")

    align_str = "Sim(3)-aligned" if not args.no_align else "unaligned"
    est_label = args.label or args.est.name
    default_title = (f"{est_label}  —  {args.metric} error  ({align_str})\n"
                     f"mean={errors.mean():.3f}{unit}  "
                     f"median={np.median(errors):.3f}{unit}  "
                     f"max={errors.max():.3f}{unit}")
    ax.set_title(args.title or default_title, fontsize=10)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
