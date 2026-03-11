
import sys
import argparse
from pathlib import Path
import torch
import numpy as np

# Add the project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.navngen.run_trajectory_lg import (
    extract_kpts_from_sequence,
    solve_poses_from_frames,
    Solver,
)
from src.navngen.filter import filter_segmentation
from src.navngen.export_trajectory import convert_tum, export_trajectory_tum, export_frames
from tqdm import tqdm

def main(args):
    """
    Runs the full visual odometry pipeline on a KITTI sequence,
    applying segmentation-based filtering to the keypoints.
    """
    # 1. Instantiate Solver
    solver = Solver(args.config_path, config_type="kitti")

    # 2. Extract keypoints and create partial frames
    partial_frames = extract_kpts_from_sequence(args.input_path)

    # 3. Load segmentation masks
    mask_files = sorted(Path(args.masks_path).glob('*.npy'))
    if len(mask_files) != len(partial_frames):
        raise ValueError(
            f"Mismatch between number of frames ({len(partial_frames)}) "
            f"and number of masks ({len(mask_files)})."
        )
    
    masks = [np.load(f) for f in tqdm(mask_files, desc="Loading masks")]

    # 4. Filter keypoints based on segmentation
    filter_ids = [int(fid) for fid in args.filter_ids.split(',')]
    print(f"Filtering frames with class IDs: {filter_ids}")
    filtered_frames = filter_segmentation(partial_frames, masks, filter_ids)

    # 5. Solve for poses using the filtered frames
    final_frames = solve_poses_from_frames(filtered_frames, solver)

    # 6. Export the trajectory and frames
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
        description="Filter a trajectory using segmentation masks and regenerate the pose estimates.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input_path", type=Path, required=True, 
                        help="Path to the KITTI sequence directory (e.g., '.../sequences/01').")
    parser.add_argument("--masks_path", type=Path, required=True, 
                        help="Path to the directory containing segmentation masks as .npy files.")
    parser.add_argument("--config_path", type=Path, required=True, 
                        help="Path to the camera calibration file (e.g., '.../sequences/01/calib.txt').")
    parser.add_argument("--filter_ids", type=str, required=True, 
                        help="""Comma-separated list of class IDs to keep for feature matching.
"
                             "Example for static features: '0,1,2,3,4,5,7,8,9,10'

"
                             "Cityscapes class mappings:
"
                             "  0: road, 1: sidewalk, 2: building, 3: wall, 4: fence,
"
                             "  5: pole, 6: traffic light, 7: traffic sign, 8: vegetation,
"
                             "  9: terrain, 10: sky, 11: person, 12: rider, 13: car,
"
                             "  14: truck, 15: bus, 16: train, 17: motorcycle, 18: bicycle.""")
    parser.add_argument("--output_path", type=Path, 
                        help="Optional: Path to save the final trajectory in TUM format.")
    parser.add_argument("--pickle_path", type=Path,
                        help="Optional: Path to save the processed frames as a compressed pickle file.")

    args = parser.parse_args()
    main(args)
