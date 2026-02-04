'''
Docstring for export_trajectory
this file exports trajectories to txt files so evo can use them
'''
from typing import Sequence
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


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
        for pose in trajectory:
            line = ' '.join(map(str, pose))
            f.write(line + '\n')

def convert_euroc_to_tum(trajectories: dict) -> np.ndarray:
    """
    Converts trajectories from the format output by gen_trajectory_euroc
    to the TUM format.

    Args:
        trajectories (dict): A dictionary mapping timestamps to (R, t) tuples,
                             where R is a 3x3 rotation matrix and t is a 3-element
                             translation vector.

    Returns:
        np.ndarray: A numpy array where each row is a pose in TUM format
                    [timestamp, tx, ty, tz, qx, qy, qz, qw].
    """
    tum_trajectory = []
    sorted_timestamps = sorted(trajectories.keys())

    for timestamp in sorted_timestamps:
        R, t = trajectories[timestamp]
        
        # Convert rotation matrix to quaternion (qx, qy, qz, qw)
        quat = Rotation.from_matrix(R).as_quat()
        
        # TUM format: timestamp tx ty tz qx qy qz qw
        pose = np.concatenate(([timestamp/1e9], t, quat))
        tum_trajectory.append(pose)

    return np.array(tum_trajectory)


def export_trajectory(trajectory, save_path="", fmt='tum'):

    pass