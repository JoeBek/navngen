#!/usr/bin/env python3
"""
plot_traj.py - plot a pickled frame trajectory

Usage:
    python3 scripts/plot_traj.py <input.pkl.gz> [--plane xz] [--label] [--title "..."] [--save out.png]
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.navngen.export_trajectory import load_frames
from src.navngen.plot import plot_trajectory
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(prog="plot_traj", description="Plot a pickled frame trajectory.")
    parser.add_argument("input", type=Path, help="Path to .pkl.gz frame file")
    parser.add_argument("--plane", choices=["xz", "xy", "yz"], default="xz", help="Projection plane (default: xz)")
    parser.add_argument("--label", action="store_true", help="Annotate each point with its frame index")
    parser.add_argument("--title", type=str, default=None, help="Plot title (default: input filename)")
    parser.add_argument("--save", type=Path, default=None, metavar="PATH", help="Save figure to PATH instead of displaying")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: file not found: {args.input}")
        sys.exit(1)

    frames = load_frames(args.input)
    title = args.title if args.title else args.input.name

    plot_trajectory(frames, plane=args.plane, label_indicies=args.label, title=title, show=False)

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
