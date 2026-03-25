
import sys
import argparse
from pathlib import Path
import torch
import numpy as np

# Add the project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.navngen.trajectory import (
    extract_kpts_from_sequence,
    solve_poses_from_frames,
    create_frame_sequence,
    Solver,
)
from src.navngen.export_trajectory import convert_tum, export_trajectory_tum, export_frames
from tqdm import tqdm

def main(args):
    """
    Runs the full visual odometry pipeline on a KITTI sequence.
    """
    # 1. Instantiate Solver
    solver = Solver(args.config_path, config_type=args.config_type)

    initial_frames = create_frame_sequence(args.input_path, args.image_dirname)

    # 2. Extract keypoints and create partial frames
    partial_frames = extract_kpts_from_sequence(initial_frames)

    # 3. Solve for poses using the frames
    final_frames = solve_poses_from_frames(partial_frames, solver)

    # 4. Export the trajectory and frames
    if final_frames:
        tum_traj = convert_tum(final_frames)
        
        # Create output directories if they don't exist
        if args.output_path:
            args.output_path.parent.mkdir(parents=True, exist_ok=True)
            export_trajectory_tum(tum_traj, args.output_path)
            print(f"Trajectory saved to {args.output_path}")

        if args.pickle_path:
            args.pickle_path.parent.mkdir(parents=True, exist_ok=True)
            export_frames(final_frames, args.pickle_path)
            print(f"Frames saved to {args.pickle_path}")
    else:
        print("No frames were processed to generate a trajectory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a trajectory from a KITTI sequence.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input_path", type=Path, required=True, 
                        help="Path to the KITTI sequence directory (e.g., '.../sequences/01').")
    parser.add_argument("--config_path", type=Path, required=True, 
                        help="Path to the camera calibration file (e.g., '.../sequences/01/calib.txt').")
    parser.add_argument("--output_path", type=Path, 
                        help="Optional: Path to save the final trajectory in TUM format.")
    parser.add_argument("--pickle_path", type=Path,
                        help="Optional: Path to save the processed frames as a compressed pickle file.")
    parser.add_argument("--config_type", type=str, default="kitti", help="type of config file to parse")
    parser.add_argument("--image_dirname", type=str, default="image_2", help="name of the directory with images in it")

    args = parser.parse_args()
    main(args)
