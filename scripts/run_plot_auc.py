#!/usr/bin/env python3
"""
run_plot_auc.py

Find every *_errors.npz under an assets directory and write an AUC@X plot
for each into a single output directory.

Usage:
    python3 scripts/run_plot_auc.py
    python3 scripts/run_plot_auc.py --threshold 10 --out-dir plots/auc
    python3 scripts/run_plot_auc.py --assets-dir assets/outputs
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]

    p = argparse.ArgumentParser()
    p.add_argument("--assets-dir", type=Path, default=project_root / "assets",
                   help="Root to search recursively for *_errors.npz")
    p.add_argument("--out-dir", type=Path,
                   default=project_root / "assets/outputs/auc_plots",
                   help="Single directory to save all PNGs")
    p.add_argument("--threshold", type=float, default=5.0,
                   help="AUC threshold in degrees (default: 5)")
    args = p.parse_args()

    plot_script = project_root / "scripts/plot_auc.py"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.assets_dir.rglob("*errors.npz"))
    if not files:
        print(f"No *_errors.npz under {args.assets_dir}")
        sys.exit(1)

    print(f"Found {len(files)} npz files  →  {args.out_dir}")

    # Detect stem collisions; for those, prefix with parent dir name.
    stems = [f.stem for f in files]
    dup = {s for s in stems if stems.count(s) > 1}

    for f in files:
        tag = f"{f.parent.name}_{f.stem}" if f.stem in dup else f.stem
        out_png = args.out_dir / f"{tag}_auc{int(args.threshold)}.png"

        print(f"\n── {f.relative_to(args.assets_dir)}")
        cmd = [
            sys.executable, str(plot_script), str(f),
            "--threshold", str(args.threshold),
            "--save-dir", str(args.out_dir),
        ]
        subprocess.run(cmd, check=True)

        # plot_auc.py names by stem; rename when we needed a collision prefix
        default_out = args.out_dir / f"{f.stem}_auc{int(args.threshold)}.png"
        if default_out != out_png and default_out.exists():
            default_out.rename(out_png)

    print(f"\nDone. Plots in {args.out_dir}")


if __name__ == "__main__":
    main()
