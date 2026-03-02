
import numpy as np
import argparse
from pathlib import Path
import sys
import gzip
import pickle
import matplotlib.pyplot as plt
from typing import Sequence

# Add the project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.navngen.frame import Frame


def load_kitti_poses(pose_file: Path) -> list:
    """Loads KITTI ground truth poses from a text file.

    Args:
        pose_file: Path to the KITTI poses file (e.g., '00.txt').

    Returns:
        A list of 4x4 numpy arrays representing the pose for each frame.
    """
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = [float(v) for v in line.strip().split()]
            pose = np.array(values).reshape(3, 4)
            pose = np.vstack((pose, [0, 0, 0, 1]))  # Convert to 4x4
            poses.append(pose)
    return poses


def calculate_relative_pose(pose1: np.ndarray, pose2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the relative pose between two absolute poses.

    Args:
        pose1: The first absolute pose (4x4 matrix).
        pose2: The second absolute pose (4x4 matrix).

    Returns:
        A tuple containing the relative rotation matrix (3x3) and
        translation vector (3x1).
    """
    relative_pose = np.linalg.inv(pose1) @ pose2
    R = relative_pose[:3, :3]
    t = relative_pose[:3, 3]
    return R, t

def plot_trial_error(errors: Sequence[float], title: str):
    """
    Plots the error for each trial.
    """
    plt.figure()
    plt.plot(errors, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("Trial Number")
    plt.ylabel("Translation Error (norm of difference)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Patch for pickle to find the 'frame' module
    if 'src.navngen.frame' in sys.modules and 'frame' not in sys.modules:
        sys.modules['frame'] = sys.modules['src.navngen.frame']
        
    parser = argparse.ArgumentParser(description="Extract pose difference for keypoint variation trials.")
    parser.add_argument("sequence", type=str, help="KITTI sequence number (e.g., '01').")
    parser.add_argument("keypoint_amount", type=int, help="Number of keypoints used in the trial (e.g., 100).")
    parser.add_argument("img_index", type=int, help="Index of the current image in the sequence for the trial.")
    parser.add_argument("--kitti_dir", type=str, default="/home/joe/data/kitti/dataset", help="Path to the KITTI dataset directory.")
    parser.add_argument("--trials_dir", type=str, default=str(project_root / "assets" / "outputs"), help="Path to the directory containing trial results.")

    args = parser.parse_args()

    # --- Load Estimated Poses from Trials ---
    trial_results_dir = Path(args.trials_dir) / f"kp_trials_{args.keypoint_amount}_idx_{args.img_index}"
    pickle_path = trial_results_dir / "result_frames.pkl.gz"

    if not pickle_path.exists():
        print(f"Error: Trial results file not found at {pickle_path}")
        sys.exit(1)

    with gzip.open(pickle_path, 'rb') as f:
        trial_frames: Sequence[Frame] = pickle.load(f)

    estimated_translations = [frame.get_essential_matrix()[1] for frame in trial_frames]

    # --- Load Ground Truth Pose ---
    pose_file = Path(args.kitti_dir) / "poses" / f"{args.sequence}.txt"

    if not pose_file.exists():
        print(f"Error: Pose file not found at {pose_file}")
        sys.exit(1)

    gt_poses = load_kitti_poses(pose_file)

    if args.img_index < 1 or args.img_index >= len(gt_poses):
        print(f"Error: img_index {args.img_index} is out of bounds for the sequence with {len(gt_poses)} poses.")
        sys.exit(1)

    pose1 = gt_poses[args.img_index - 1]
    pose2 = gt_poses[args.img_index]

    gt_R, gt_t = calculate_relative_pose(pose1, pose2)
    
    # --- Calculate Error ---
    # The estimated translation is a unit vector. Normalize the ground truth for comparison.
    gt_t_unit = gt_t / np.linalg.norm(gt_t)

    errors = [np.linalg.norm(est_t - gt_t_unit) for est_t in estimated_translations]

    print(f"Ground Truth Relative Translation (Unit Vector):\n{gt_t_unit}\n")
    print(f"Mean Estimated Relative Translation:\n{np.mean(estimated_translations, axis=0)}\n")
    print(f"Mean Error: {np.mean(errors)}")
    print(f"Std Dev of Error: {np.std(errors)}")


    # --- Plot Error ---
    plot_title = (f"Relative Translation Error vs. Ground Truth\n"
                  f"Sequence {args.sequence}, Images {args.img_index-1}-{args.img_index}, {args.keypoint_amount} Keypoints")
    plot_trial_error(errors, plot_title)
