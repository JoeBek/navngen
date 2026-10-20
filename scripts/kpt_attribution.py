#!/usr/bin/env python3
"""
kpt_attribution.py

Per-keypoint depth and segmentation attribution for a filtered frame pickle.

For each keypoint that survived filtering, record:
  - its (x, y) image coordinate
  - the depth value already stored on the frame (frame.kpt_depth)
  - the segmentation class ID looked up at the keypoint location from the
    matching mask in --seg-masks

Output is a single .npz with a flat per-keypoint table plus per-frame counts.

Usage:
    python3 scripts/kpt_attribution.py <frames.pkl.gz> <output.npz> \\
        --seg-masks <dir>
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.navngen.export_trajectory import load_frames


def kpt_seg_classes(kpts: np.ndarray, seg_map: np.ndarray) -> np.ndarray:
    """Sample seg class ID at each keypoint (x, y). Out-of-bounds → -1."""
    xy = np.floor(kpts).astype(np.int64)
    h, w = seg_map.shape
    x, y = xy[:, 0], xy[:, 1]
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.full(len(kpts), -1, dtype=np.int32)
    out[valid] = seg_map[y[valid], x[valid]].astype(np.int32)
    return out


def build_parser():
    p = argparse.ArgumentParser(prog="kpt_attribution")
    p.add_argument("frames",     type=Path, help="Input .pkl.gz frame file")
    p.add_argument("output",     type=Path, help="Output .npz file")
    p.add_argument("--seg-masks", type=Path, required=True,
                   help="Directory of segmentation masks (*.npy, names match image stems)")
    return p


def main():
    args = build_parser().parse_args()

    print(f"Loading frames: {args.frames}")
    frames = load_frames(args.frames)
    print(f"  {len(frames)} frames")

    frame_idx      = []
    kpt_xy_all     = []
    kpt_depth_all  = []
    kpt_seg_all    = []
    per_frame_n    = np.zeros(len(frames), dtype=np.int32)
    missing_mask   = 0

    for i, f in enumerate(tqdm(frames, desc="Attributing keypoints")):
        if f.kpts is None or f.kpt_depth is None or f.path is None:
            continue

        kpts  = np.asarray(f.kpts.cpu().numpy() if isinstance(f.kpts, torch.Tensor)
                           else f.kpts, dtype=np.float32)
        depth = np.asarray(f.kpt_depth.cpu().numpy() if isinstance(f.kpt_depth, torch.Tensor)
                           else f.kpt_depth, dtype=np.float32)

        mask_path = args.seg_masks / (Path(f.path).stem + ".npy")
        if not mask_path.exists():
            missing_mask += 1
            seg = np.full(len(kpts), -1, dtype=np.int32)
        else:
            seg_map = np.load(mask_path)
            seg     = kpt_seg_classes(kpts, seg_map)

        n = len(kpts)
        per_frame_n[i] = n
        frame_idx.append(np.full(n, i, dtype=np.int32))
        kpt_xy_all.append(kpts)
        kpt_depth_all.append(depth)
        kpt_seg_all.append(seg)

    if missing_mask:
        print(f"  WARNING: {missing_mask} frames had no matching seg mask "
              f"(seg=-1 for their keypoints)")

    frame_indices = np.concatenate(frame_idx)        if frame_idx     else np.zeros(0, np.int32)
    kpt_xy        = np.concatenate(kpt_xy_all)       if kpt_xy_all    else np.zeros((0, 2), np.float32)
    kpt_depth     = np.concatenate(kpt_depth_all)    if kpt_depth_all else np.zeros(0, np.float32)
    kpt_seg       = np.concatenate(kpt_seg_all)      if kpt_seg_all   else np.zeros(0, np.int32)

    print(f"\nTotal keypoints: {len(kpt_depth)}  over {(per_frame_n > 0).sum()} frames")
    if len(kpt_depth):
        print(f"  depth   mean={kpt_depth.mean():.4f}  median={np.median(kpt_depth):.4f}  "
              f"min={kpt_depth.min():.4f}  max={kpt_depth.max():.4f}")
        classes, counts = np.unique(kpt_seg, return_counts=True)
        print(f"  seg classes (id: count):")
        for c, n in zip(classes.tolist(), counts.tolist()):
            print(f"    {c:>3d}: {n}")

    np.savez(
        args.output,
        frame_indices = frame_indices,
        kpt_xy        = kpt_xy,
        kpt_depth     = kpt_depth,
        kpt_seg       = kpt_seg,
        per_frame_n   = per_frame_n,
    )
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
