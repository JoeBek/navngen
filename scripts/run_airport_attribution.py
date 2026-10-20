#!/usr/bin/env python3
"""
run_airport_attribution.py

Run inlier-set attribution (eval_inliers.py) on every airport frame pickle
found under assets/outputs/airport_frames/, comparing against the airport
ground-truth TUM file.

Output: one *_inlier_errors.npz per trial in
    assets/outputs/subsample_errors/airport/

These npz files plot directly with scripts/plot_auc.py and
scripts/plot_subsample_errors.py.

Usage:
    python3 scripts/run_airport_attribution.py
    python3 scripts/run_airport_attribution.py --trials airport_MSNV_..._00156
"""

import sys
import argparse
import subprocess
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

FRAMES_DIR = project_root / "assets/outputs/airport_frames"
GT_FILE    = project_root / "assets/outputs/airport/groundtruth.txt"
CALIB      = Path("/home/joe/vt/research/glue/gluetest/configs/camera_forward.yaml")
OUT_DIR    = project_root / "assets/outputs/subsample_errors/airport"


def build_parser():
    p = argparse.ArgumentParser(prog="run_airport_attribution")
    p.add_argument("--trials", type=str, nargs="+", default=None,
                   help="Frame stems to process (default: all *_both_frames.pkl.gz)")
    p.add_argument("--gt", type=Path, default=GT_FILE,
                   help=f"GT TUM file (default: {GT_FILE})")
    p.add_argument("--calib", type=Path, default=CALIB,
                   help=f"Camera YAML config (default: {CALIB})")
    return p


def main():
    args = build_parser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_script = project_root / "scripts/eval_inliers.py"
    all_frames  = sorted(FRAMES_DIR.glob("*_both_frames.pkl.gz"))

    if args.trials:
        wanted = set(args.trials)
        frames_paths = [p for p in all_frames
                        if p.stem.replace(".pkl", "") in wanted
                        or p.name.replace("_both_frames.pkl.gz", "") in wanted]
    else:
        frames_paths = all_frames

    if not frames_paths:
        print(f"No frame pickles found in {FRAMES_DIR}")
        sys.exit(1)

    print(f"Found {len(frames_paths)} trial(s) to evaluate")
    for path in frames_paths:
        tag = path.name.replace("_both_frames.pkl.gz", "")
        out_npz = OUT_DIR / f"{tag}_inlier_errors.npz"

        print(f"\n── {tag} ────────────────────────────────────────")
        cmd = [
            sys.executable, str(eval_script),
            str(path), str(args.gt), str(out_npz),
            "--config_path", str(args.calib),
            "--camera",      "surfnav",
            "--gt_format",   "tum",
        ]
        subprocess.run(cmd, check=True)
        print(f"  → {out_npz}")

    print(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
