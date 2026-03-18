#!/usr/bin/env python3
import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Evaluate KITTI trajectories with evo_ape")
    parser.add_argument("input_dir", type=str, help="Directory containing estimated trajectories")
    parser.add_argument("--gt_dir", type=str, default="assets/outputs/kitti/gt", help="Directory containing GT trajectories")
    parser.add_argument("--plot_dir", type=str, default="assets/plots", help="Directory to save plots")
    parser.add_argument("--num_seq", type=int, default=10, help="Number of sequences to evaluate (default 10 for 00-09)")
    parser.add_argument("--plot", action="store_true", help="Show plot window (equivalent to -p)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    gt_dir = Path(args.gt_dir)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(args.num_seq), desc="Evaluating trajectories"):
        seq = f"{i:02d}"
        gt_file = gt_dir / f"{seq}.txt"
        
        if not gt_file.exists():
            # Using tqdm.write to avoid interfering with the progress bar
            tqdm.write(f"Skipping seq {seq}, GT file {gt_file} not found")
            continue
            
        # Try to find the estimated file: seq_traj.kitti, seq.txt, seq.kitti
        # Prioritize _traj.kitti as in user example
        est_file = None
        for suffix in ["_traj.kitti", ".txt", ".kitti"]:
            test_file = input_dir / f"{seq}{suffix}"
            if test_file.exists():
                est_file = test_file
                break
        
        if not est_file:
            tqdm.write(f"Skipping seq {seq}, estimated file not found in {input_dir}")
            continue

        # Save plot with the sequence name to avoid overwriting
        # The user mentioned assets/plots/ so we append the filename
        output_plot = plot_dir / f"{seq}_ape_xz.pdf"

        # Construct command based on user's request:
        # evo_ape kitti 00.txt 00_traj.kitti -p -plot_mode xz -as -save_plot assets/plots/
        # Note: -as corresponds to -a (align) and -s (correct scale)
        # We use --save_plot with a full filename to be safe.
        cmd = [
            "evo_ape", "kitti",
            str(gt_file),
            str(est_file),
            "-a", "-s",
            "--plot_mode", "xz",
            "--save_plot", str(output_plot),
            "--silent" # To keep output clean
        ]
        
        if args.plot:
            cmd.append("-p")

        try:
            # We run it and let evo handle the heavy lifting
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            tqdm.write(f"Error evaluating sequence {seq}: {e}")

if __name__ == "__main__":
    main()
