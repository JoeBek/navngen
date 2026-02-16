
import numpy as np
from typing import Sequence, Tuple
from .frame import Frame


def get_frame_index_by_second(second:int, fps:int) -> int:

    return second * fps

def get_subtrajectory(frames:Sequence[Frame], t_start, t_end, fps) -> Sequence[Frame]:
    
    start_index = get_frame_index_by_second(t_start, fps)
    end_index = get_frame_index_by_second(t_end, fps)
    
    frame_list = list(frames)
    return frame_list[start_index:end_index]

def rotation_matrix_to_euler_angles(R: np.ndarray, degrees: bool = False) -> Tuple[float, float, float]:
    """
    Converts a 3x3 rotation matrix to Euler angles (roll, pitch, yaw).
    Assumes ZYX (yaw, pitch, roll) intrinsic rotation order.

    Args:
        R (np.ndarray): A 3x3 rotation matrix.
        degrees (bool): If True, returns angles in degrees. Otherwise, in radians.

    Returns:
        Tuple[float, float, float]: (roll, pitch, yaw) angles.
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    if degrees:
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    else:
        return roll, pitch, yaw
        
