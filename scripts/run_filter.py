import argparse
from pathlib import Path

import numpy as np

# Modular keypoint-filtering stage. Consumes SuperPoint feature .npz files
# (run_superpoint.py) plus per-frame depth and/or segmentation masks, prunes the
# keypoints/descriptors, and writes filtered .npz files in the SAME schema so the
# output feeds straight into run_lightglue.py --features_path.
#
# Numpy-only port of src/navngen/filter.py (filter_depth / filter_segmentation)
# so this stage needs no torch/lightglue -> tiny image.

FEATURE_KEYS = ("keypoints", "keypoint_scores", "descriptors", "image_size")


def sample_at_kpts(mask_map: np.ndarray, kpts: np.ndarray, invalid):
    """Sample a (H,W) map at integer-floored keypoint (x,y) locations.
    Returns (values, valid_mask). Out-of-bounds kpts get `invalid` and valid=False."""
    idx = np.floor(kpts).astype(int)
    x, y = idx[:, 0], idx[:, 1]
    h, w = mask_map.shape[:2]
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    values = np.full(len(kpts), invalid, dtype=np.float64)
    values[valid] = mask_map[y[valid], x[valid]]
    return values, valid


def main(args):
    feat_files = sorted(args.features_path.glob("*.npz"))
    if not feat_files:
        print(f"No feature .npz files under {args.features_path}")
        return

    use_depth = args.mode in ("depth", "both")
    use_seg = args.mode in ("seg", "both")

    depth_files = seg_files = None
    if use_depth:
        depth_files = sorted(args.depth_path.glob("*.npy"))
        if len(depth_files) != len(feat_files):
            raise ValueError(
                f"Count mismatch: {len(feat_files)} feature files vs "
                f"{len(depth_files)} depth maps.")
    if use_seg:
        seg_files = sorted(args.seg_path.glob("*.npy"))
        if len(seg_files) != len(feat_files):
            raise ValueError(
                f"Count mismatch: {len(feat_files)} feature files vs "
                f"{len(seg_files)} segmentation masks.")

    filter_ids = None
    if use_seg:
        filter_ids = [int(x) for x in args.filter_ids.split(",")]

    # Load all features once (keypoints are small). Keypoint depths are sampled
    # here too so normalization can span the whole sequence before thresholding.
    frames = []
    for i, f in enumerate(feat_files):
        data = {k: np.load(f, allow_pickle=True)[k] for k in np.load(f).files}
        kpts = data["keypoints"]
        depth_vals = depth_valid = None
        if use_depth:
            dmap = np.load(depth_files[i])
            depth_vals, depth_valid = sample_at_kpts(dmap, kpts, invalid=-1.0)
        frames.append({"file": f, "data": data, "kpts": kpts,
                       "depth_vals": depth_vals, "depth_valid": depth_valid})

    # Optional global depth normalization (matches normalize_depth): map all valid
    # keypoint depths to [0,1] before applying tl/th thresholds.
    if use_depth and args.normalize:
        alld = np.concatenate([fr["depth_vals"][fr["depth_valid"]] for fr in frames
                               if fr["depth_valid"].any()]) if frames else np.array([])
        if alld.size:
            dmin, dmax = float(alld.min()), float(alld.max())
            scale = dmax - dmin
            for fr in frames:
                v = fr["depth_vals"]
                if scale > 0:
                    norm = (v - dmin) / scale
                    norm[~fr["depth_valid"]] = -1.0  # keep invalid as invalid
                    fr["depth_vals"] = norm
                else:
                    fr["depth_vals"] = np.where(fr["depth_valid"], 0.0, -1.0)

    args.output_path.mkdir(parents=True, exist_ok=True)
    total_in = total_out = 0

    for i, fr in enumerate(frames):
        n = len(fr["kpts"])
        keep = np.ones(n, dtype=bool)

        if use_depth:
            d = fr["depth_vals"]
            dkeep = fr["depth_valid"] & (d >= args.tl)
            if args.th > args.tl:
                dkeep &= (d <= args.th)
            keep &= dkeep

        if use_seg:
            smap = np.load(seg_files[i])
            cls, svalid = sample_at_kpts(smap, fr["kpts"], invalid=-1.0)
            skeep = svalid & np.isin(cls.astype(int), filter_ids)
            keep &= skeep

        data = fr["data"]
        out = dict(data)
        for k in FEATURE_KEYS:
            if k == "image_size":
                continue
            if k in out:
                out[k] = out[k][keep]

        np.savez_compressed(args.output_path / fr["file"].name, **out)
        total_in += n
        total_out += int(keep.sum())

    print(f"Filtered {len(frames)} frames ({args.mode}): "
          f"{total_in} -> {total_out} keypoints kept "
          f"({100.0 * total_out / max(total_in, 1):.1f}%). Saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter SuperPoint keypoints by depth and/or segmentation "
        "masks and write filtered features for LightGlue.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--features_path", type=Path, required=True,
                        help="Directory of SuperPoint .npz files (run_superpoint.py output).")
    parser.add_argument("--output_path", type=Path, required=True,
                        help="Directory to write filtered .npz files.")
    parser.add_argument("--mode", choices=["depth", "seg", "both"], required=True,
                        help="Which filters to apply.")
    parser.add_argument("--depth_path", type=Path,
                        help="Directory of depth maps (*.npy), required for depth/both.")
    parser.add_argument("--seg_path", type=Path,
                        help="Directory of segmentation class-id maps (*.npy), required for seg/both.")
    parser.add_argument("--tl", type=float, default=0.0, help="Lower depth threshold.")
    parser.add_argument("--th", type=float, default=0.0,
                        help="Upper depth threshold (ignored if <= tl).")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize keypoint depths to [0,1] across the sequence before thresholding.")
    parser.add_argument("--filter_ids", type=str, default="0,1,2,3,4,5,7,8,9",
                        help="Comma-separated Cityscapes class IDs to KEEP (seg/both).")

    args = parser.parse_args()
    if args.mode in ("depth", "both") and not args.depth_path:
        parser.error("--depth_path is required for mode depth/both")
    if args.mode in ("seg", "both") and not args.seg_path:
        parser.error("--seg_path is required for mode seg/both")
    main(args)
