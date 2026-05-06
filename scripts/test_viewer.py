"""
test_viewer.py — replay a saved pickle through the viewer.

Loads pre-computed frames from a .pkl.gz file, triangulates between
consecutive frame pairs, and pushes each frame to the live viewer.
No GPU required.

Usage:
    python scripts/test_viewer.py
    python scripts/test_viewer.py --pickle assets/outputs/01_pickle.pkl.gz \
                                  --calib  assets/sequences/01/calib.txt \
                                  --delay  0.05
"""

import sys
import time
import argparse
from pathlib import Path

# allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from navngen.export_trajectory import load_frames
from navngen.triangulate import triangulate_frame_pair, K_from_camera
from navngen.camera import parse_camera_kitti
from navngen.viewer import Viewer
from navngen.frame import Frame


def main():
    repo = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Replay a pickle through the navngen viewer.")
    parser.add_argument(
        "--pickle",
        default=str(repo / "assets/outputs/01_pickle.pkl.gz"),
        help="Path to a .pkl.gz file produced by export_frames().",
    )
    parser.add_argument(
        "--calib",
        default=str(repo / "assets/sequences/01/calib.txt"),
        help="Path to a KITTI calib.txt (or omit to use a fallback K).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Seconds to wait between frames (default 0.05 = 20 fps replay).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames (default: all).",
    )
    args = parser.parse_args()

    pickle_path = Path(args.pickle)
    calib_path  = Path(args.calib)

    # ── load frames ──────────────────────────────────────────────────────────
    print(f"Loading frames from {pickle_path} ...")
    frames = load_frames(pickle_path)
    print(f"  {len(frames)} frames loaded.")

    if args.max_frames is not None:
        frames = frames[: args.max_frames]
        print(f"  Capped to {len(frames)} frames.")

    # ── camera intrinsics ────────────────────────────────────────────────────
    if calib_path.exists():
        camera = parse_camera_kitti(calib_path)
        K = K_from_camera(camera)
        print(f"  K loaded from {calib_path}")
    else:
        import numpy as np
        print(f"  WARNING: calib not found at {calib_path}, using fallback K.")
        # rough KITTI-like defaults
        K = np.array([[718.856, 0., 607.193],
                      [0., 718.856, 185.216],
                      [0., 0., 1.]], dtype=float)

    # ── start viewer ─────────────────────────────────────────────────────────
    print("Opening viewer windows ...")
    viewer = Viewer()

    # push frame 0 (no matches yet)
    viewer.update(frames[0])
    time.sleep(args.delay)

    # ── replay loop ───────────────────────────────────────────────────────────
    print("Replaying — close the viewer windows or Ctrl+C to stop.")
    try:
        for i in range(1, len(frames)):
            pts_3d = triangulate_frame_pair(frames[i - 1], frames[i], K)
            viewer.update(frames[i], pts_3d)
            time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nInterrupted.")

    print("Done — keeping viewer open. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    viewer.stop()
    viewer.join()


if __name__ == "__main__":
    main()
