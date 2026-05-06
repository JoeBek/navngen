#!/usr/bin/env python3
"""
frametool.py - inspect pickled frame sequences

Usage:
    python3 frametool.py <input.pkl.gz> <frame_number> <command> [argument] [--flags]

Commands:
    show keypoints          display the frame image with keypoints overlaid
    show matches            display the frame image with matches to the next frame
    info length             print the number of frames in the sequence
    info pose               print relative pose of frame N vs frame N-1 (frame 0 vs identity)

Video flags (render all frames without re-loading the pickle):
    --save_video OUT.mp4    encode every frame to a video
    --fps N                 frames per second (default: 10)
    --start N               first frame index (default: 0)
    --end N                 last frame index inclusive (default: last)
"""

import sys
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.navngen.export_trajectory import load_frames
from src.navngen.load_images import load_image
import matplotlib
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_show(frames, frame_idx, argument, args):
    frame = frames[frame_idx]
    path = frame.get_path()
    image = load_image(path)

    from lightglue import viz2d

    if argument == "keypoints":
        viz2d.plot_images([image])
        kpts = frame.get_kpts()
        viz2d.plot_keypoints([kpts], ps=6)
        plt.suptitle(f"Frame {frame_idx} — keypoints ({len(kpts)})")

    elif argument == "matches":
        next_idx = frame_idx + 1
        if next_idx >= len(frames):
            raise ValueError(f"frame {frame_idx} is the last frame, no next frame for matches")
        next_frame = frames[next_idx]
        next_path = next_frame.get_path()
        image1 = load_image(next_path)

        matches = next_frame.get_matches()
        kpts0 = frame.get_kpts()
        kpts1 = next_frame.get_kpts()

        m_kpts0 = kpts0[matches[:, 0]]
        m_kpts1 = kpts1[matches[:, 1]]

        if args.inliers:
            import matplotlib.colors as mcolors
            inlier_mask = next_frame.get_info()["inliers"]
            c_in = mcolors.to_rgba("lime")
            c_out = mcolors.to_rgba("red")
            colors = [c_in if inlier_mask[i] else c_out for i in range(len(matches))]
            n_inliers = sum(bool(v) for v in inlier_mask)
            title = f"Matches: frame {frame_idx} → {next_idx} ({n_inliers}/{len(matches)} inliers)"
        else:
            colors = "lime"
            title = f"Matches: frame {frame_idx} → {next_idx} ({len(matches)})"

        viz2d.plot_images([image, image1])
        viz2d.plot_matches(m_kpts0, m_kpts1, color=colors, lw=0.2)
        plt.suptitle(title)

    else:
        print(f"Unknown argument for 'show': {argument!r}")
        print("Valid arguments: keypoints, matches")
        sys.exit(1)

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _describe_unit_translation(t):
    """Human-readable direction from a unit translation vector (camera frame: X=right, Y=down, Z=forward)."""
    t = np.asarray(t, dtype=float)
    norm = np.linalg.norm(t)
    if norm < 1e-6:
        return "none (zero translation)"
    t = t / norm

    x, y, z = t
    components = sorted(
        [(abs(x), x, "right", "left"),
         (abs(y), y, "down",  "up"),
         (abs(z), z, "forward", "backward")],
        reverse=True,
    )
    threshold = 0.3
    parts = []
    for i, (mag, val, pos_lbl, neg_lbl) in enumerate(components):
        if i == 0 or mag > threshold:
            parts.append(pos_lbl if val >= 0 else neg_lbl)
    return "-".join(parts)


def _describe_rotation(R):
    """Return (angle_deg, axis_vec_str, axis_desc, yaw_deg, pitch_deg, roll_deg)."""
    from scipy.spatial.transform import Rotation as Rot
    rot = Rot.from_matrix(R)
    angle_rad = rot.magnitude()
    angle_deg = float(np.degrees(angle_rad))

    if angle_rad < 1e-6:
        return angle_deg, "[0  0  0]", "no rotation", 0.0, 0.0, 0.0

    axis = rot.as_rotvec() / angle_rad
    ax, ay, az = axis
    dominant = int(np.argmax(np.abs(axis)))
    axis_desc_map = {
        0: "X-axis (pitch: nose up/down)",
        1: "Y-axis (yaw: left/right)",
        2: "Z-axis (roll: clockwise/counter-clockwise)",
    }
    axis_vec_str = f"[{ax:+.3f}  {ay:+.3f}  {az:+.3f}]"

    # Decompose as yaw (Y), pitch (X), roll (Z) — intrinsic YXZ in camera convention
    yaw, pitch, roll = rot.as_euler("YXZ", degrees=True)
    return angle_deg, axis_vec_str, axis_desc_map[dominant], float(yaw), float(pitch), float(roll)


def cmd_info(frames, frame_idx, argument, args):
    if argument == "length":
        print(len(frames))

    elif argument == "pose":
        frame = frames[frame_idx]
        prev_label = "identity" if frame_idx == 0 else f"frame {frame_idx - 1}"

        print(f"Frame {frame_idx} — relative pose vs {prev_label}")
        print("─" * 50)

        if frame_idx == 0 or not hasattr(frame, "E") or frame.E is None:
            R_rel = np.eye(3)
            t_unit = np.zeros(3)
            if frame_idx == 0:
                print("  (first frame — pose is the identity reference)")
        else:
            R_rel, t_unit = frame.E
            t_unit = np.asarray(t_unit, dtype=float)

        t_unit = np.asarray(t_unit, dtype=float)
        norm = np.linalg.norm(t_unit)

        print()
        print("Translation (unit direction in previous camera frame)")
        print(f"  vector : [{t_unit[0]:+.4f}  {t_unit[1]:+.4f}  {t_unit[2]:+.4f}]")
        print(f"  |t|    : {norm:.6f}")
        print(f"  approx : {_describe_unit_translation(t_unit)}")

        print()
        angle_deg, axis_str, axis_desc, yaw, pitch, roll = _describe_rotation(R_rel)
        print("Rotation (relative to previous frame)")
        print(f"  angle  : {angle_deg:.3f}°")
        print(f"  axis   : {axis_str}  ≈ {axis_desc}")
        print(f"  yaw    : {yaw:+.3f}°  (left/right)")
        print(f"  pitch  : {pitch:+.3f}°  (up/down)")
        print(f"  roll   : {roll:+.3f}°  (tilt)")

    else:
        print(f"Unknown argument for 'info': {argument!r}")
        print("Valid arguments: length, pose")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Command registry — add new commands here
# ---------------------------------------------------------------------------

COMMANDS = {
    "show": cmd_show,
    "info": cmd_info,
}


# ---------------------------------------------------------------------------
# Video rendering (iterates all frames in one process)
# ---------------------------------------------------------------------------

def render_video(frames, args):
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found in PATH")
        sys.exit(1)

    matplotlib.use("Agg")

    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else len(frames) - 1

    if start < 0 or end >= len(frames) or start > end:
        print(f"Error: frame range [{start}, {end}] invalid for sequence of length {len(frames)}")
        sys.exit(1)

    handler = COMMANDS[args.command]
    tmp_dir = Path(tempfile.mkdtemp(prefix="frametool_video_"))

    try:
        rendered = []
        total = end - start + 1
        skipped = 0

        for i, frame_idx in enumerate(range(start, end + 1)):
            out_path = tmp_dir / f"frame_{frame_idx:06d}.png"
            args.save = out_path
            try:
                handler(frames, frame_idx, args.argument, args)
                rendered.append(out_path)
            except Exception as e:
                skipped += 1
            print(f"\r  {i + 1}/{total} rendered, {skipped} skipped", end="", flush=True)

        print()

        if not rendered:
            print("Error: no frames rendered successfully")
            sys.exit(1)

        list_file = tmp_dir / "frames.txt"
        with list_file.open("w") as f:
            for p in sorted(rendered):
                f.write(f"file '{p}'\n")
                f.write(f"duration {1 / args.fps}\n")

        print(f"Encoding {len(rendered)} frames → {args.save_video}")
        result = subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(args.save_video),
        ])
        if result.returncode != 0:
            print("Error: ffmpeg encoding failed")
            sys.exit(1)

    finally:
        shutil.rmtree(tmp_dir)

    print(f"Done: {args.save_video}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="frametool",
        description="Inspect pickled frame sequences.",
    )
    parser.add_argument("input", type=Path, help="Path to .pkl.gz frame file")
    parser.add_argument("frame", type=int, nargs="?", default=None, help="Frame index (0-based); ignored when --save_video is used")
    parser.add_argument("command", choices=list(COMMANDS), help="Command to run")
    parser.add_argument("argument", nargs="?", default=None, help="Argument for the command")
    parser.add_argument("--inliers", action="store_true", help="Color RANSAC inliers green and outliers red (show matches only)")
    parser.add_argument("--save", type=Path, default=None, metavar="PATH", help="Save figure to PATH instead of displaying it")
    parser.add_argument("--save_video", type=Path, default=None, metavar="PATH", help="Render all frames to a video file")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for --save_video (default: 10)")
    parser.add_argument("--start", type=int, default=None, help="First frame index for --save_video (default: 0)")
    parser.add_argument("--end", type=int, default=None, help="Last frame index for --save_video (default: last)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: file not found: {args.input}")
        sys.exit(1)

    frames = load_frames(args.input)

    if args.save_video:
        render_video(frames, args)
        return

    if args.frame is None:
        parser.error("frame index is required when not using --save_video")

    if args.frame < 0 or args.frame >= len(frames):
        print(f"Error: frame index {args.frame} out of range [0, {len(frames) - 1}]")
        sys.exit(1)

    handler = COMMANDS[args.command]
    handler(frames, args.frame, args.argument, args)


if __name__ == "__main__":
    main()
