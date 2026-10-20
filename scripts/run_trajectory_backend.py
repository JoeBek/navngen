import sys
import argparse
import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from poselib import estimate_relative_pose

# Project root (scripts/ -> repo root)
project_root = Path(__file__).resolve().parents[1]

# RANSAC options mirror src/navngen/trajectory.py:Solver so the modular backend
# produces the same relative poses as the monolithic get_trajectory pipeline.
RANSAC_OPTIONS = {
    "max_epipolar_error": 0.5,
    "min_inlier_ratio": 0.5,
    "confidence": 0.999,
    "max_iterations": 50000,
    "min_iterations": 1000,
}


def _load_camera_module():
    """
    Load src/navngen/camera.py directly, bypassing the navngen package __init__
    (which imports torch/lightglue/open3d). The backend only needs intrinsics
    parsing, keeping its dependency set to numpy/scipy/pyyaml/poselib.
    """
    cam_path = project_root / "src" / "navngen" / "camera.py"
    spec = importlib.util.spec_from_file_location("navngen_camera", cam_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


camera_mod = _load_camera_module()


def parse_camera(config_path: Path, config_type: str) -> dict:
    if config_type == "surfnav":
        return camera_mod.parse_camera_surfnav(config_path)
    if config_type == "euroc":
        return camera_mod.parse_camera_euroc(config_path)
    if config_type == "kitti":
        return camera_mod.parse_camera_kitti(config_path)
    raise ValueError(f"Unknown config_type: {config_type}")


def compose_with_unit_direction(R_prev, t_prev, R_rel, u_rel, scale=1.0):
    """Mirror of trajectory.compose_with_unit_direction (pure-numpy, inlined to
    avoid importing the heavy navngen.trajectory module)."""
    u = np.asarray(u_rel, dtype=float).reshape(3)
    norm = np.linalg.norm(u)
    t_rel = u / norm * float(scale) if norm > 0 else np.zeros(3, dtype=float)
    R_new = R_prev @ R_rel
    t_new = t_prev + R_prev @ t_rel
    return R_new, t_new


def build_K(config_path: Path, config_type: str) -> np.ndarray:
    """Build the 3x3 intrinsics matrix from the calibration file. Reads the raw
    file (not the poselib param vector) so the fx/fy/cx/cy order is correct for
    every config_type."""
    import yaml
    if config_type == "surfnav":
        with open(config_path) as f:
            data = np.asarray(yaml.safe_load(f)["camera_matrix"]["data"], dtype=float)
        return data.reshape(3, 3)
    if config_type == "euroc":
        with open(config_path) as f:
            fx, fy, cx, cy = (float(v) for v in yaml.safe_load(f)["intrinsics"])
    elif config_type == "kitti":
        with open(config_path) as f:
            for line in f:
                if line.startswith("P0:"):
                    p = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
                    fx, fy, cx, cy = p[0, 0], p[1, 1], p[0, 2], p[1, 2]
                    break
            else:
                raise ValueError("Could not find P0 in calibration file")
    else:
        raise ValueError(f"Unknown config_type: {config_type}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def _sample_depth(depth_map: np.ndarray, kpts: np.ndarray):
    """Sample a depth map at floored keypoint (x,y). Returns (depths, valid)."""
    idx = np.floor(kpts).astype(int)
    x, y = idx[:, 0], idx[:, 1]
    h, w = depth_map.shape[:2]
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    d = np.zeros(len(kpts), dtype=float)
    d[valid] = depth_map[y[valid], x[valid]]
    return d, valid


def estimate_metric_scale(m_kpts0, m_kpts1, R_rel, t_rel, depth0, depth1, Kinv):
    """Recover a metric translation scale for one pair from per-frame depth.

    Back-projects matched keypoints to metric 3D using each frame's depth, then
    finds the scale s such that s * unit(t_rel) best matches the depth-implied
    baseline (X1 = R_rel X0 + t_metric). Robust via the median over inliers.
    Returns None if no usable points.
    """
    d0, v0 = _sample_depth(depth0, m_kpts0)
    d1, v1 = _sample_depth(depth1, m_kpts1)
    valid = v0 & v1 & (d0 > 0) & (d1 > 0)
    if valid.sum() < 3:
        return None

    def backproject(kpts, depths):
        uv1 = np.concatenate([kpts, np.ones((len(kpts), 1))], axis=1)  # (N,3)
        rays = (Kinv @ uv1.T).T                                        # (N,3)
        return rays * depths[:, None]

    X0 = backproject(m_kpts0[valid], d0[valid])
    X1 = backproject(m_kpts1[valid], d1[valid])
    t_est = X1 - (R_rel @ X0.T).T  # (N,3) depth-implied translation per point

    tn = np.linalg.norm(t_rel)
    if tn == 0:
        return None
    unit_t = np.asarray(t_rel, dtype=float).reshape(3) / tn
    s = float(np.median(t_est @ unit_t))
    if not np.isfinite(s) or s <= 0:
        s = float(np.median(np.linalg.norm(t_est, axis=1)))
    return s if np.isfinite(s) and s > 0 else None


def load_pairs(matches_path: Path) -> list[dict]:
    """Load LightGlue match .npz files (from run_lightglue.py), sorted by name."""
    files = sorted(matches_path.glob("*.npz"))
    pairs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        pairs.append({
            "file": f.name,
            "name0": str(d["name0"]),
            "name1": str(d["name1"]),
            "matches": d["matches"],
            "kpts0": d["keypoints0"],
            "kpts1": d["keypoints1"],
        })
    return pairs


def build_frame_order(pairs: list[dict]) -> list[str]:
    """Chain consecutive pairs into an ordered list of frame names."""
    order = [pairs[0]["name0"]]
    for p in pairs:
        if p["name0"] != order[-1]:
            raise ValueError(
                f"Non-consecutive match chain: pair {p['file']} starts at "
                f"'{p['name0']}' but previous frame was '{order[-1]}'. "
                "Match files must form a continuous frame chain."
            )
        order.append(p["name1"])
    return order


def load_timestamps(times_path: Path, n_frames: int) -> list[float]:
    """Read one timestamp per line, aligned to the sorted frame order."""
    with times_path.open() as f:
        stamps = [float(line) for line in f if line.strip()]
    if len(stamps) < n_frames:
        print(
            f"Warning: times file has {len(stamps)} entries but {n_frames} frames; "
            "trajectory will be truncated to the shorter length."
        )
    return stamps


def main(args):
    pairs = load_pairs(args.matches_path)
    if not pairs:
        print(f"No match .npz files found under {args.matches_path}")
        return

    frame_order = build_frame_order(pairs)
    camera = parse_camera(args.config_path, args.config_type)

    # Timestamps: real values from --times_path, else integer frame indices.
    if args.times_path:
        timestamps = load_timestamps(args.times_path, len(frame_order))
    else:
        timestamps = list(range(len(frame_order)))

    # Optional metric scaling: per-pair translation scale from depth maps,
    # aligned to the sorted frame order.
    depth_files = Kinv = None
    if args.scale_mode == "metric":
        depth_files = sorted(args.depth_path.glob("*.npy"))
        if len(depth_files) < len(frame_order):
            raise ValueError(
                f"metric scaling needs >= {len(frame_order)} depth maps, "
                f"found {len(depth_files)} in {args.depth_path}.")
        Kinv = np.linalg.inv(build_K(args.config_path, args.config_type))

    # Chain relative poses into absolute poses (first frame = identity).
    R, t = np.eye(3), np.zeros(3)
    poses = [(R, t)]
    for j, p in enumerate(pairs):
        matches = p["matches"]
        if matches.shape[0] == 0:
            raise ValueError(f"Pair {p['file']} has zero matches; cannot solve pose.")
        m_kpts0 = p["kpts0"][matches[:, 0]]
        m_kpts1 = p["kpts1"][matches[:, 1]]

        pose, _info = estimate_relative_pose(
            m_kpts0, m_kpts1, camera, camera, RANSAC_OPTIONS
        )

        scale = args.scale
        if args.scale_mode == "metric":
            d0 = np.load(depth_files[j])
            d1 = np.load(depth_files[j + 1])
            s = estimate_metric_scale(m_kpts0, m_kpts1, pose.R, pose.t, d0, d1, Kinv)
            scale = args.scale * s if s is not None else args.scale

        R, t = compose_with_unit_direction(R, t, pose.R, pose.t, scale=scale)
        poses.append((R, t))

    # Export TUM: timestamp tx ty tz qx qy qz qw (timestamps passed through as-is).
    n = min(len(timestamps), len(poses))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w") as f:
        for (R, t), stamp in zip(poses[:n], timestamps[:n]):
            quat = Rotation.from_matrix(R).as_quat()  # (qx, qy, qz, qw)
            row = [stamp, *t.tolist(), *quat.tolist()]
            f.write(" ".join(map(str, row)) + "\n")

    print(f"Trajectory saved to {args.output_path} ({n} poses)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trajectory backend: consume LightGlue match .npz files "
        "(from run_lightglue.py), estimate relative poses with poselib, chain "
        "them into an absolute trajectory, and export it in TUM format.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--matches_path", type=Path, required=True,
                        help="Directory of LightGlue match .npz files (run_lightglue.py output).")
    parser.add_argument("--config_path", type=Path, required=True,
                        help="Camera calibration file (same as get_trajectory.py).")
    parser.add_argument("--output_path", type=Path, required=True,
                        help="Path to write the trajectory in TUM format.")
    parser.add_argument("--config_type", type=str, default="kitti",
                        help="Calibration format: kitti | surfnav | euroc.")
    parser.add_argument("--times_path", type=Path,
                        help="Optional: file with one timestamp per line, aligned to the\n"
                             "sorted frame order. Defaults to integer frame indices.")
    parser.add_argument("--scale_mode", choices=["unit", "metric"], default="unit",
                        help="unit: unit-norm translations (default). metric: recover a\n"
                             "per-pair metric scale from depth maps (--depth_path).")
    parser.add_argument("--depth_path", type=Path,
                        help="Directory of per-frame depth maps (*.npy), required for metric.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Global multiplier applied to every translation (default 1.0).")

    args = parser.parse_args()
    if args.scale_mode == "metric" and not args.depth_path:
        parser.error("--depth_path is required for --scale_mode metric")
    main(args)
