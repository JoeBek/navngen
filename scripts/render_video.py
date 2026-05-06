#!/usr/bin/env python3
"""
render_video.py - render a frametool show command across all frames into a video

Usage:
    python3 scripts/render_video.py <input.pkl.gz> <command> [argument] [frametool flags] --output video.mp4

Examples:
    python3 scripts/render_video.py assets/outputs/foo.pkl.gz show matches --inliers --output out.mp4
    python3 scripts/render_video.py assets/outputs/foo.pkl.gz show keypoints --output out.mp4 --fps 15
    python3 scripts/render_video.py assets/outputs/foo.pkl.gz show matches --start 100 --end 300 --output out.mp4
"""

import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path


FRAMETOOL = Path(__file__).resolve().parent / "frametool.py"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="render_video",
        description="Render frametool show command for every frame and encode to video.",
    )
    parser.add_argument("input", type=Path, help="Path to .pkl.gz frame file")
    parser.add_argument("command", help="frametool command (e.g. show)")
    parser.add_argument("argument", nargs="?", default=None, help="Argument for the command (e.g. matches, keypoints)")
    parser.add_argument("--output", type=Path, required=True, help="Output video path (e.g. out.mp4)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second in output video (default: 10)")
    parser.add_argument("--start", type=int, default=None, help="First frame index (default: 0)")
    parser.add_argument("--end", type=int, default=None, help="Last frame index inclusive (default: last frame)")
    parser.add_argument("--keep-frames", action="store_true", help="Keep rendered frame images after encoding")
    # pass-through flags for frametool
    parser.add_argument("--inliers", action="store_true", help="Pass --inliers to frametool")
    return parser


def get_frame_count(input_path: Path) -> int:
    result = subprocess.run(
        [sys.executable, str(FRAMETOOL), str(input_path), "0", "info", "length"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error getting frame count:\n{result.stderr}")
        sys.exit(1)
    return int(result.stdout.strip())


def render_frame(input_path: Path, frame_idx: int, command: str, argument: str | None,
                 extra_flags: list[str], out_path: Path) -> bool:
    cmd = [sys.executable, str(FRAMETOOL), str(input_path), str(frame_idx), command]
    if argument is not None:
        cmd.append(argument)
    cmd += extra_flags
    cmd += ["--save", str(out_path)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return True


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: file not found: {args.input}")
        sys.exit(1)

    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found in PATH")
        sys.exit(1)

    n_frames = get_frame_count(args.input)
    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else n_frames - 1

    if start < 0 or end >= n_frames or start > end:
        print(f"Error: frame range [{start}, {end}] invalid for sequence of length {n_frames}")
        sys.exit(1)

    extra_flags = []
    if args.inliers:
        extra_flags.append("--inliers")

    tmp_dir = Path(tempfile.mkdtemp(prefix="frametool_render_"))
    print(f"Rendering frames to {tmp_dir}")

    rendered = []
    skipped = 0
    total = end - start + 1

    for i, frame_idx in enumerate(range(start, end + 1)):
        out_path = tmp_dir / f"frame_{frame_idx:06d}.png"
        ok = render_frame(args.input, frame_idx, args.command, args.argument, extra_flags, out_path)
        if ok:
            rendered.append(out_path)
        else:
            skipped += 1
        print(f"\r  {i + 1}/{total} frames rendered, {skipped} skipped", end="", flush=True)

    print()

    if not rendered:
        print("Error: no frames rendered successfully")
        shutil.rmtree(tmp_dir)
        sys.exit(1)

    # Create a sorted file list for ffmpeg to handle gaps from skipped frames
    list_file = tmp_dir / "frames.txt"
    with list_file.open("w") as f:
        for p in sorted(rendered):
            f.write(f"file '{p}'\n")
            f.write(f"duration {1 / args.fps}\n")

    print(f"Encoding {len(rendered)} frames → {args.output}")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # ensure even dimensions for h264
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(args.output),
    ]
    result = subprocess.run(ffmpeg_cmd)
    if result.returncode != 0:
        print("Error: ffmpeg encoding failed")
        shutil.rmtree(tmp_dir)
        sys.exit(1)

    if args.keep_frames:
        print(f"Frame images kept at {tmp_dir}")
    else:
        shutil.rmtree(tmp_dir)

    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
