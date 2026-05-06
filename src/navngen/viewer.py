"""
Viewer — live 3D + 2D visualization for navngen.

Two windows:
  "navngen: Map Viewer"    — open3d 3D window: point cloud, camera frustums,
                             trajectory line.
  "navngen: Current Frame" — OpenCV 2D window: current image with feature
                             overlays (green = matched, blue = unmatched).

Usage:
    viewer = Viewer()
    # inside your VO loop, after each frame is processed:
    viewer.update(frame, new_points_3d)   # new_points_3d may be None
    # when done:
    viewer.stop()
    viewer.join()
"""

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .frame import Frame

try:
    import open3d as o3d
    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False


# ── shared state ──────────────────────────────────────────────────────────────

class _ViewerState:
    def __init__(self):
        self.lock = threading.Lock()
        # accumulated data
        self.poses: list = []                                  # list of (R, t)
        self.map_points: np.ndarray = np.empty((0, 3), dtype=np.float64)
        # per-frame overlay data
        self.current_image: Optional[np.ndarray] = None       # grayscale uint8
        self.kpts: Optional[np.ndarray] = None                 # (N, 2) float
        self.match_flags: Optional[np.ndarray] = None          # (N,) bool
        self.n_poses: int = 0
        # control
        self.dirty: bool = False
        self.running: bool = True


# ── geometry helpers ───────────────────────────────────────────────────────────

def _frustum_lines(R: np.ndarray, t: np.ndarray, size: float = 0.05):
    """
    Returns (points, line_indices, colors) for a camera frustum in world space.
    R = Rwc (3x3), t = camera center in world (3,).
    """
    w, h, z = size, size * 0.75, size * 0.6
    # 5 points: origin + 4 image-plane corners (in camera space)
    cam_pts = np.array([
        [0,  0,  0 ],
        [w,  h,  z ],
        [w, -h,  z ],
        [-w, -h,  z ],
        [-w,  h,  z ],
    ], dtype=np.float64)
    world_pts = (R @ cam_pts.T).T + t  # (5, 3)

    lines = [[0,1],[0,2],[0,3],[0,4],   # four rays
             [1,2],[2,3],[3,4],[4,1]]   # image-plane rectangle
    colors = [[0.0, 0.0, 1.0]] * 8     # blue
    return world_pts, lines, colors


def _build_frustum_lineset(poses, size=0.05):
    """
    Build a single open3d LineSet for all keyframe frustums.
    Only called when new poses have arrived.
    """
    if not _HAS_OPEN3D:
        return None
    all_pts, all_lines, all_colors = [], [], []
    offset = 0
    for R, t in poses:
        pts, lines, colors = _frustum_lines(
            np.asarray(R, dtype=np.float64),
            np.asarray(t, dtype=np.float64),
            size=size
        )
        all_pts.append(pts)
        all_lines.extend([[i + offset, j + offset] for i, j in lines])
        all_colors.extend(colors)
        offset += len(pts)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    ls.lines  = o3d.utility.Vector2iVector(np.array(all_lines))
    ls.colors = o3d.utility.Vector3dVector(np.array(all_colors))
    return ls


def _draw_frame_overlay(
    image: np.ndarray,
    kpts: Optional[np.ndarray],
    match_flags: Optional[np.ndarray],
    n_poses: int,
    n_points: int,
) -> np.ndarray:
    """
    Annotate a grayscale image with keypoint overlays and a status bar.
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    r = 5
    if kpts is not None and match_flags is not None:
        for i, (x, y) in enumerate(kpts):
            color = (0, 255, 0) if match_flags[i] else (255, 0, 0)
            xi, yi = int(x), int(y)
            cv2.rectangle(bgr, (xi - r, yi - r), (xi + r, yi + r), color, 1)
            cv2.circle(bgr, (xi, yi), 2, color, -1)

    # status bar
    h, w = bgr.shape[:2]
    bar = np.zeros((22, w, 3), dtype=np.uint8)
    n_matched = int(np.sum(match_flags)) if match_flags is not None else 0
    status = f"Poses: {n_poses}  Points: {n_points}  Matched: {n_matched}"
    cv2.putText(bar, status, (5, 16), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bgr, bar])


# ── viewer ─────────────────────────────────────────────────────────────────────

class Viewer:
    """
    Live viewer with a 3D map window (open3d) and a 2D frame window (OpenCV).
    Runs in a daemon thread; call update() from your VO loop.
    """

    def __init__(self, map_title: str = "navngen: Map Viewer",
                 frame_title: str = "navngen: Current Frame",
                 frustum_size: float = 0.05,
                 fps: float = 30.0):
        if not _HAS_OPEN3D:
            raise ImportError(
                "open3d is required for Viewer. Install it with: pip install open3d"
            )
        self._map_title    = map_title
        self._frame_title  = frame_title
        self._frustum_size = frustum_size
        self._period       = 1.0 / fps
        self._state        = _ViewerState()
        self._thread       = threading.Thread(target=self._run, daemon=True, name="navngen-viewer")
        self._thread.start()

    # ── public API ─────────────────────────────────────────────────────────────

    def update(self, frame: Frame, new_points: Optional[np.ndarray] = None) -> None:
        """
        Push new data to the viewer.  Thread-safe; call from your VO loop.

        Args:
            frame:      The just-processed Frame (must have .pose set).
            new_points: (M, 3) array of newly triangulated world-space points,
                        or None.
        """
        # Load image from disk before taking the lock (slow I/O outside mutex)
        image = None
        if frame.path is not None:
            image = cv2.imread(str(frame.path), cv2.IMREAD_GRAYSCALE)

        kpts = None
        if frame.kpts is not None:
            k = frame.kpts
            kpts = k.numpy() if hasattr(k, 'numpy') else np.asarray(k)

        match_flags = None
        if kpts is not None and frame.matches is not None:
            m = frame.matches
            m = m.numpy() if hasattr(m, 'numpy') else np.asarray(m)
            flags = np.zeros(len(kpts), dtype=bool)
            if m.ndim == 2 and m.shape[1] >= 2:
                idx1 = m[:, 1]
                idx1 = idx1[idx1 < len(kpts)]
                flags[idx1] = True
            match_flags = flags

        with self._state.lock:
            if frame.pose is not None:
                R, t = frame.pose
                self._state.poses.append(
                    (np.asarray(R, dtype=np.float64),
                     np.asarray(t, dtype=np.float64))
                )
                self._state.n_poses = len(self._state.poses)

            if new_points is not None and len(new_points) > 0:
                pts = np.asarray(new_points, dtype=np.float64)
                self._state.map_points = np.vstack([self._state.map_points, pts])

            if image is not None:
                self._state.current_image = image
            if kpts is not None:
                self._state.kpts = kpts
            if match_flags is not None:
                self._state.match_flags = match_flags

            self._state.dirty = True

    def stop(self) -> None:
        """Signal the viewer thread to shut down."""
        with self._state.lock:
            self._state.running = False

    def join(self, timeout: float = 5.0) -> None:
        """Wait for the viewer thread to finish."""
        self._thread.join(timeout=timeout)

    # ── render loop ────────────────────────────────────────────────────────────

    def _run(self) -> None:
        # ── open3d setup ──
        vis = o3d.visualization.Visualizer()
        vis.create_window(self._map_title, width=1024, height=768)
        opt = vis.get_render_option()
        opt.background_color = np.array([1.0, 1.0, 1.0])
        opt.point_size = 2.0
        opt.line_width = 1.5

        # seed with a dummy point so the bounding box is never empty
        _dummy = o3d.geometry.PointCloud()
        _dummy.points = o3d.utility.Vector3dVector(np.array([[0., 0., 0.]]))
        vis.add_geometry(_dummy)

        pcd = o3d.geometry.PointCloud()
        traj_ls = o3d.geometry.LineSet()
        frustum_ls = o3d.geometry.LineSet()

        vis.add_geometry(pcd)
        vis.add_geometry(traj_ls)
        vis.add_geometry(frustum_ls)

        # ── opencv setup ──
        try:
            cv2.namedWindow(self._frame_title, cv2.WINDOW_NORMAL)
            _cv2_ok = True
        except Exception:
            _cv2_ok = False

        prev_n_poses  = 0
        prev_n_points = 0

        while True:
            # ── snapshot state ──
            with self._state.lock:
                running = self._state.running
                dirty   = self._state.dirty
                if dirty:
                    poses      = list(self._state.poses)
                    map_pts    = self._state.map_points.copy()
                    cur_img    = (self._state.current_image.copy()
                                  if self._state.current_image is not None else None)
                    kpts       = (self._state.kpts.copy()
                                  if self._state.kpts is not None else None)
                    mflags     = (self._state.match_flags.copy()
                                  if self._state.match_flags is not None else None)
                    n_poses    = self._state.n_poses
                    self._state.dirty = False

            if not running:
                break

            if dirty:
                n_pts = len(map_pts)

                # ── point cloud ──
                if n_pts > prev_n_points:
                    pcd.points = o3d.utility.Vector3dVector(map_pts)
                    pcd.paint_uniform_color([0.15, 0.15, 0.75])
                    vis.update_geometry(pcd)
                    prev_n_points = n_pts

                # ── trajectory line ──
                if len(poses) >= 2:
                    traj_pts = np.array([t for _, t in poses])
                    traj_ls.points = o3d.utility.Vector3dVector(traj_pts)
                    traj_ls.lines  = o3d.utility.Vector2iVector(
                        [[i, i + 1] for i in range(len(traj_pts) - 1)]
                    )
                    traj_ls.paint_uniform_color([0.0, 0.7, 0.0])
                    vis.update_geometry(traj_ls)

                # ── frustums (rebuild when new poses arrive) ──
                if n_poses > prev_n_poses and len(poses) > 0:
                    new_ls = _build_frustum_lineset(poses, size=self._frustum_size)
                    if new_ls is not None:
                        frustum_ls.points = new_ls.points
                        frustum_ls.lines  = new_ls.lines
                        frustum_ls.colors = new_ls.colors
                        vis.update_geometry(frustum_ls)
                    prev_n_poses = n_poses
                    vis.reset_view_point(True)

                # ── 2D image overlay ──
                if _cv2_ok and cur_img is not None:
                    annotated = _draw_frame_overlay(
                        cur_img, kpts, mflags, n_poses, n_pts
                    )
                    try:
                        cv2.imshow(self._frame_title, annotated)
                    except Exception:
                        pass

            vis.poll_events()
            vis.update_renderer()
            if _cv2_ok:
                cv2.waitKey(1)

            time.sleep(self._period)

        # ── cleanup ──
        vis.destroy_window()
        if _cv2_ok:
            cv2.destroyWindow(self._frame_title)
