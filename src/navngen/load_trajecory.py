import numpy as np

class TrajectoryPoint:
    def __init__(self, timestamp, x, y, z, qx, qy, qz, qw):
        self.timestamp = timestamp
        self.position = np.array([x, y, z])
        self.orientation = np.array([qx, qy, qz, qw]) # Quaternion (x, y, z, w)

    def __repr__(self):
        return (f"TrajectoryPoint(timestamp={self.timestamp}, "
                f"position={self.position}, orientation={self.orientation})")

def load_trajectory_from_txt(file_path):
    """
    Loads trajectory data from a text file.

    Each line in the file is expected to contain:
    timestamp x y z qx qy qz qw

    Args:
        file_path (str): The path to the text file.

    Returns:
        list[TrajectoryPoint]: A list of TrajectoryPoint objects.
    """
    trajectory_points = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
            try:
                parts = list(map(float, line.split()))
                if len(parts) == 8:
                    timestamp, x, y, z, qx, qy, qz, qw = parts
                    trajectory_points.append(TrajectoryPoint(timestamp, x, y, z, qx, qy, qz, qw))
                else:
                    print(f"Warning: Skipping malformed line (expected 8 values): {line}")
            except ValueError as e:
                print(f"Warning: Skipping line due to parsing error: {line} - {e}")
    return trajectory_points




if __name__ == '__main__':
    # Example usage: Create a dummy trajectory file and load it
    dummy_file_path = "/home/joe/nav_2/ORB_SLAM3/Examples/Monocular/CameraTrajectory.txt"
    print(f"Loading trajectory from {dummy_file_path}...")
    trajectory = load_trajectory_from_txt(dummy_file_path)

    for point in trajectory:
        print(point)

    