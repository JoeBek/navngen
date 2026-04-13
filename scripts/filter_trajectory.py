
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
    create_frame_sequence_euroc,
    Solver,
)
from src.navngen.filter import filter_segmentation, filter_depth
from src.navngen.export_trajectory import convert_tum, export_trajectory_tum, export_frames
from tqdm import tqdm

def get_kpt_depth(kpts: torch.Tensor, depth_map: np.ndarray) -> torch.Tensor:
    """Samples depth for each keypoint."""
    # kpts are (x, y), so we need to convert to integer indices.
    kpt_indices = torch.floor(kpts).long()
    
    # Ensure indices are within depth map bounds
    height, width = depth_map.shape
    x_indices, y_indices = kpt_indices[:, 0], kpt_indices[:, 1]
    
    valid_mask = (x_indices >= 0) & (x_indices < width) & (y_indices >= 0) & (y_indices < height)
    
    # Initialize depths with a value indicating invalidity (e.g., -1 or NaN)
    # Using -1 as a placeholder for out-of-bounds keypoints.
    depths = torch.full((len(kpts),), -1.0, dtype=torch.float32)

    # Get depths for valid keypoints.
    # In numpy, indexing is (row, col) which corresponds to (y, x).
    valid_depths = depth_map[y_indices[valid_mask], x_indices[valid_mask]]
    
    depths[valid_mask] = torch.from_numpy(valid_depths.astype(np.float32))
    
    return depths

def main(args):
    """
    Runs the full visual odometry pipeline on a KITTI sequence,
    applying filtering to the keypoints.
    """
    # 1. Instantiate Solver
    solver = Solver(args.config_path, config_type=args.config_type)

    # 2. Extract keypoints and create partial frames

    if args.config_type == 'euroc':
        initial_frames = create_frame_sequence_euroc(args.input_path)
    else:
        initial_frames = create_frame_sequence(args.input_path, args.image_dirname)
    # 2. Extract keypoints and create partial frames
    partial_frames = extract_kpts_from_sequence(initial_frames)

    # 4. Filter keypoints based on the selected mode
    if args.filter_mode == 'segmentation':
        # Load segmentation masks
        mask_files = sorted(Path(args.mask_path).glob('*.npy'))
        if len(mask_files) != len(partial_frames):
            raise ValueError(
                f"Mismatch between number of frames ({len(partial_frames)}) "
                f"and number of masks ({len(mask_files)})."
            )

        filter_ids = [int(fid) for fid in args.filter_ids.split(',')]
        print(f"Filtering frames with class IDs: {filter_ids}")
        mask_gen = (np.load(f) for f in tqdm(mask_files, desc="Filtering segmentation"))
        filtered_frames = filter_segmentation(partial_frames, mask_gen, filter_ids)

    elif args.filter_mode == 'depth':
        # Load depth maps
        depth_files = sorted(Path(args.mask_path).glob('*.npy'))
        if len(depth_files) != len(partial_frames):
            raise ValueError(
                f"Mismatch between number of frames ({len(partial_frames)}) "
                f"and number of depth maps ({len(depth_files)})."
            )

        # Assign depth to keypoints one file at a time to avoid loading all depth maps at once
        for frame, depth_file in tqdm(zip(partial_frames, depth_files), desc="Assigning depths",
                                      total=len(depth_files)):
            if frame.kpts is not None:
                depth_map = np.load(depth_file)
                frame.kpt_depth = get_kpt_depth(frame.kpts.cpu(), depth_map).to(frame.kpts.device)

        print(f"Filtering frames with depth thresholds: tl={args.tl}, th={args.th}")
        filtered_frames = filter_depth(partial_frames, args.normalize, tl=args.tl, th=args.th)

    elif args.filter_mode == 'both':
        # --- depth pass: load one depth map at a time ---
        depth_files = sorted(Path(args.depth_mask_path).glob('*.npy'))
        if len(depth_files) != len(partial_frames):
            raise ValueError(
                f"Mismatch between number of frames ({len(partial_frames)}) "
                f"and number of depth maps ({len(depth_files)})."
            )
        for frame, depth_file in tqdm(zip(partial_frames, depth_files), desc="Assigning depths",
                                      total=len(depth_files)):
            if frame.kpts is not None:
                depth_map = np.load(depth_file)
                frame.kpt_depth = get_kpt_depth(frame.kpts.cpu(), depth_map).to(frame.kpts.device)
        print(f"Filtering frames with depth thresholds: tl={args.tl}, th={args.th}")
        depth_filtered = filter_depth(partial_frames, args.normalize, tl=args.tl, th=args.th)

        # --- segmentation pass on the depth-filtered frames ---
        seg_files = sorted(Path(args.seg_mask_path).glob('*.npy'))
        if len(seg_files) != len(depth_filtered):
            raise ValueError(
                f"Mismatch between number of frames ({len(depth_filtered)}) "
                f"and number of seg masks ({len(seg_files)})."
            )
        filter_ids = [int(fid) for fid in args.filter_ids.split(',')]
        print(f"Filtering frames with class IDs: {filter_ids}")
        seg_gen = (np.load(f) for f in tqdm(seg_files, desc="Filtering segmentation"))
        filtered_frames = filter_segmentation(depth_filtered, seg_gen, filter_ids)

    else:
        raise ValueError(f"Unknown filter mode: {args.filter_mode}")


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
        description="Filter a trajectory using segmentation masks or depth maps and regenerate the pose estimates.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # General arguments
    parser.add_argument("--input-path", "-i", type=Path, required=True, 
                        help="Path to the KITTI sequence directory (e.g., '.../sequences/01').")
    parser.add_argument("--config_path", "-c", type=Path, required=True, 
                        help="Path to the camera calibration file (e.g., '.../sequences/01/calib.txt').")
    parser.add_argument("--output_path", "-o", type=Path, 
                        help="Optional: Path to save the final trajectory in TUM format.")
    parser.add_argument("--pickle_path", "-p", type=Path,
                        help="Optional: Path to save the processed frames as a compressed pickle file.")
    parser.add_argument("--image_dirname", type=str, default="image_2", help="name of the directory with images in it")
    parser.add_argument("--config_type", type=str, default="kitti", help="type of config file to parse")

    # Filtering mode
    filter_parsers = parser.add_subparsers(dest='filter_mode', required=True, help="Filtering mode")

    # Segmentation filtering arguments
    seg_parser = filter_parsers.add_parser('segmentation', help="Filter keypoints using segmentation masks.")
    seg_parser.add_argument("--mask-path", "-m", type=Path, required=True, 
                        help="Path to the directory containing segmentation masks as .npy files.")
    seg_parser.add_argument("--filter-ids", type=str, required=True, 
                        help='''Comma-separated list of class IDs to keep for feature matching.
Example for static features: '0,1,2,3,4,5,7,8,9,10'
Cityscapes class mappings:
  0: road, 1: sidewalk, 2: building, 3: wall, 4: fence,
  5: pole, 6: traffic light, 7: traffic sign, 8: vegetation,
  9: terrain, 10: sky, 11: person, 12: rider, 13: car,
  14: truck, 15: bus, 16: train, 17: motorcycle, 18: bicycle.''')

    # Depth filtering arguments
    depth_parser = filter_parsers.add_parser('depth', help="Filter keypoints using depth maps.")
    depth_parser.add_argument("--mask-path", "-m", type=Path, required=True,
                              help="Path to the directory containing depth maps as .npy files.")
    depth_parser.add_argument("--tl", type=float, required=True, help="Lower depth threshold.")
    depth_parser.add_argument("--th", type=float, required=True, help="Upper depth threshold.")
    depth_parser.add_argument("--normalize", action="store_true", help="Normalize the depth points after filtering.")

    # Combined depth + segmentation filtering arguments
    both_parser = filter_parsers.add_parser('both', help="Apply depth filtering then segmentation filtering.")
    both_parser.add_argument("--depth-mask-path", type=Path, required=True,
                             help="Path to the directory containing depth maps as .npy files.")
    both_parser.add_argument("--seg-mask-path", type=Path, required=True,
                             help="Path to the directory containing segmentation masks as .npy files.")
    both_parser.add_argument("--tl", type=float, required=True, help="Lower depth threshold.")
    both_parser.add_argument("--th", type=float, required=True, help="Upper depth threshold.")
    both_parser.add_argument("--normalize", action="store_true", help="Normalize depth points after filtering.")
    both_parser.add_argument("--filter-ids", type=str, required=True,
                             help="Comma-separated list of segmentation class IDs to keep.")

    args = parser.parse_args()
    main(args)
