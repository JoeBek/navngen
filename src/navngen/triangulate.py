import numpy as np
import cv2
from .frame import Frame


def K_from_camera(camera: dict) -> np.ndarray:
    """
    Extract a 3x3 intrinsic matrix from the poselib camera dict.
    Expected params layout: [fx, fy, cx, cy, k1, k2, ...]
    """
    p = camera['params']
    fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    return np.array([[fx, 0., cx],
                     [0., fy, cy],
                     [0., 0., 1.]], dtype=np.float64)


def _projection_matrix(R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Build a 3x4 projection matrix P = K @ [R^T | -R^T t].

    Convention used throughout navngen:
      pose = (R, t) where R = Rwc (camera orientation in world) and
      t = camera center in world.
    The world-to-camera transform is therefore:
      p_cam = R^T @ (p_world - t)
    """
    R_cw = R.T
    t_cw = -(R.T @ t)
    return K @ np.hstack([R_cw, t_cw.reshape(3, 1)])


def triangulate_frame_pair(
    frame_prev: Frame,
    frame_curr: Frame,
    K: np.ndarray,
) -> np.ndarray:
    """
    Triangulate matched keypoints between two consecutive frames.

    Args:
        frame_prev: Previous frame, must have .pose and .kpts set.
        frame_curr: Current frame, must have .pose, .kpts, and .matches set.
                    matches shape: (N, 2) where col-0 indexes frame_prev.kpts
                    and col-1 indexes frame_curr.kpts.
        K:          3x3 camera intrinsic matrix.

    Returns:
        (M, 3) float64 array of world-space 3D points.
        May be empty if inputs are invalid or all points are behind cameras.
    """
    if frame_prev.pose is None or frame_curr.pose is None:
        return np.empty((0, 3), dtype=np.float64)
    if frame_curr.matches is None or frame_curr.kpts is None or frame_prev.kpts is None:
        return np.empty((0, 3), dtype=np.float64)

    matches = frame_curr.matches
    if hasattr(matches, 'numpy'):
        matches = matches.numpy()
    matches = np.asarray(matches)
    if matches.ndim != 2 or matches.shape[1] < 2 or len(matches) == 0:
        return np.empty((0, 3), dtype=np.float64)

    kpts0 = frame_prev.kpts
    kpts1 = frame_curr.kpts
    if hasattr(kpts0, 'numpy'):
        kpts0 = kpts0.numpy()
    if hasattr(kpts1, 'numpy'):
        kpts1 = kpts1.numpy()
    kpts0 = np.asarray(kpts0, dtype=np.float64)
    kpts1 = np.asarray(kpts1, dtype=np.float64)

    idx0 = matches[:, 0]
    idx1 = matches[:, 1]

    # Guard against out-of-range indices
    valid = (idx0 < len(kpts0)) & (idx1 < len(kpts1))
    idx0, idx1 = idx0[valid], idx1[valid]
    if len(idx0) == 0:
        return np.empty((0, 3), dtype=np.float64)

    m0 = kpts0[idx0].T  # (2, N)
    m1 = kpts1[idx1].T  # (2, N)

    R0, t0 = frame_prev.pose
    R1, t1 = frame_curr.pose

    P0 = _projection_matrix(np.asarray(R0, dtype=np.float64),
                             np.asarray(t0, dtype=np.float64), K)
    P1 = _projection_matrix(np.asarray(R1, dtype=np.float64),
                             np.asarray(t1, dtype=np.float64), K)

    pts_4d = cv2.triangulatePoints(P0, P1, m0, m1)  # (4, N)
    w = pts_4d[3]
    good_w = np.abs(w) > 1e-8
    pts_4d = pts_4d[:, good_w]
    w = w[good_w]
    pts_3d = (pts_4d[:3] / w).T  # (M, 3)

    if len(pts_3d) == 0:
        return np.empty((0, 3), dtype=np.float64)

    # Keep only points with positive depth in both cameras
    def _depth(R, t, pts):
        R_cw = np.asarray(R, dtype=np.float64).T
        t_cw = -(R_cw @ np.asarray(t, dtype=np.float64))
        return (R_cw @ pts.T + t_cw.reshape(3, 1)).T[:, 2]

    d0 = _depth(R0, t0, pts_3d)
    d1 = _depth(R1, t1, pts_3d)
    keep = (d0 > 0) & (d1 > 0)
    return pts_3d[keep]
