'''
Docstring for export_trajectory
this file exports trajectories to txt files so evo can use them
'''
from typing import Sequence
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from frame import Frame
import pickle
import gzip
from tqdm import tqdm


def export_trajectory_tum(trajectory, save_path: Path):
    """
    Exports a trajectory in TUM format to a text file.

    The expected format for each pose in the trajectory is a numpy array
    containing [timestamp, tx, ty, tz, qx, qy, qz, qw].

    Args:
        trajectory (Sequence[np.ndarray]): A sequence of poses.
        save_path (Path): The path to save the trajectory file.
    """
    with save_path.open('w') as f:
        for pose in tqdm(trajectory, desc="exporting to txt in TUM format..."):
            line = ' '.join(map(str, pose))
            f.write(line + '\n')

def convert_tum(frames: Sequence[Frame]) -> np.ndarray:
    """
    Converts a sequence of Frame objects to the TUM trajectory format.

    Args:
        frames (Sequence[Frame]): A sequence of Frame objects, where each Frame
                                   contains a timestamp and a pose (R, t).

    Returns:
        np.ndarray: A numpy array where each row is a pose in TUM format
                    [timestamp, tx, ty, tz, qx, qy, qz, qw].
    """
    tum_trajectory = []

    for frame in frames:
        R, t = frame.get_pose()
        timestamp = frame.timestamp
        
        # Convert rotation matrix to quaternion (qx, qy, qz, qw)
        quat = Rotation.from_matrix(R).as_quat()
        
        # TUM format: timestamp tx ty tz qx qy qz qw
        # Convert timestamp to seconds (assuming it's in nanoseconds)
        pose = np.concatenate(([timestamp / 1e9], t, quat))
        tum_trajectory.append(pose)

    return np.array(tum_trajectory)

def export_frames(frames: Sequence[Frame], save_path: Path):
    """
    Exports a sequence of Frame objects to a compressed file using pickle and gzip.

    Args:
        frames (Sequence[Frame]): A sequence of Frame objects to export.
        save_path (Path): The path to save the compressed file (e.g., 'frames.pkl.gz').
    """
    with gzip.open(save_path, 'wb') as f:
        pickle.dump(frames, f)


def load_frames(load_path: Path) -> Sequence[Frame]:
    """
    Loads a sequence of Frame objects from a compressed file using pickle and gzip.

    Args:
        load_path (Path): The path to the compressed file (e.g., 'frames.pkl.gz').

    Returns:
        Sequence[Frame]: The loaded sequence of Frame objects.
    """
    with gzip.open(load_path, 'rb') as f:
        frames = pickle.load(f)
    return frames