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
"""

import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
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
    return p


def main():
    args = build_parser().parse_args()

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

    kpts0   = to_numpy(f_prev.kpts)   # (P, 2) — all kpts in frame i
    kpts1   = to_numpy(f_curr.kpts)   # (Q, 2) — all kpts in frame j
    matches = to_numpy(f_curr.matches) # (M, 2) — index pairs into kpts0/kpts1

    n_matches = len(matches)
    n_sample  = min(args.N, n_matches)

    if n_sample < 5:
        print(f"Error: only {n_matches} matches — need at least 5")
        sys.exit(1)

    print(f"Pair ({i}, {j}): {n_matches} matches, subsampling N={n_sample} over K={args.K} trials")

    # Accumulate selected keypoint coords across all trials
    selected0 = []  # pixel coords in frame i
    selected1 = []  # pixel coords in frame j

    for _ in range(args.K):
        idx  = np.random.choice(n_matches, size=n_sample, replace=False)
        sub  = matches[idx]
        selected0.append(kpts0[sub[:, 0]])
        selected1.append(kpts1[sub[:, 1]])

    selected0 = np.concatenate(selected0, axis=0)  # (K*N, 2)
    selected1 = np.concatenate(selected1, axis=0)

    # Load images
    img0 = image_tensor_to_rgb(load_image(f_prev.path))
    img1 = image_tensor_to_rgb(load_image(f_curr.path))
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    hmap0 = build_heatmap(selected0, h0, w0, args.sigma)
    hmap1 = build_heatmap(selected1, h1, w1, args.sigma)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Keypoint subsample heatmap  —  N={args.N}, K={args.K} trials\n"
        f"frames: {i} (left)  /  {j} (right)  |  {n_matches} total matches",
        fontsize=11,
    )

    kpts_all0 = kpts0[matches[:, 0]] if args.show_kpts else None
    kpts_all1 = kpts1[matches[:, 1]] if args.show_kpts else None

    overlay_heatmap(ax0, img0, hmap0, args.alpha, args.cmap,
                    f"Frame {i} — prev", kpts_all0)
    overlay_heatmap(ax1, img1, hmap1, args.alpha, args.cmap,
                    f"Frame {j} — curr", kpts_all1)

    # Shared colourbar
    sm = plt.cm.ScalarMappable(cmap=args.cmap,
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=[ax0, ax1], fraction=0.025, pad=0.02,
                 label="relative selection frequency")

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
