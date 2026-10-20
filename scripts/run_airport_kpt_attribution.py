#!/usr/bin/env python3
"""
run_airport_kpt_attribution.py

Run kpt_attribution.py on every airport frame pickle, using the seg masks
in the matching staging directory.

Outputs go to assets/outputs/kpt_attribution/airport/
"""

import sys
import argparse
import subprocess
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

FRAMES_DIR  = project_root / "assets/outputs/airport_frames"
SEG_MASKS   = project_root / "assets/staging/airport/seg_masks"
OUT_DIR     = project_root / "assets/outputs/kpt_attribution/airport"


def build_parser():
    p = argparse.ArgumentParser(prog="run_airport_kpt_attribution")
    p.add_argument("--trials", type=str, nargs="+", default=None,
                   help="Tags to process (default: all *_both_frames.pkl.gz)")
    p.add_argument("--seg-masks", type=Path, default=SEG_MASKS,
                   help=f"Seg masks dir (default: {SEG_MASKS})")
    return p


def main():
    args = build_parser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    script = project_root / "scripts/kpt_attribution.py"
    all_frames = sorted(FRAMES_DIR.glob("*_both_frames.pkl.gz"))

    if args.trials:
        wanted = set(args.trials)
        frames_paths = [p for p in all_frames
                        if p.name.replace("_both_frames.pkl.gz", "") in wanted]
    else:
        frames_paths = all_frames

    if not frames_paths:
        print(f"No frame pickles in {FRAMES_DIR}")
        sys.exit(1)

    print(f"Processing {len(frames_paths)} trial(s)")
    for path in frames_paths:
        tag = path.name.replace("_both_frames.pkl.gz", "")
        out = OUT_DIR / f"{tag}_kpt_attribution.npz"

        print(f"\n── {tag} ────────────────────────────────────────")
        subprocess.run([
            sys.executable, str(script),
            str(path), str(out),
            "--seg-masks", str(args.seg_masks),
        ], check=True)
        print(f"  → {out}")

    print(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
