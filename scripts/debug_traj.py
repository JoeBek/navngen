
import sys
from pathlib import Path
from lightglue import viz2d

# Add the project root to sys.path
# This assumes the script is run from within the project structure
project_root = Path(__file__).resolve().parents[1] # Go up one level from debug_traj.py to /home/joe/vt/research/glue/navngen/
sys.path.insert(0, str(project_root))

from src.navngen.export_trajectory import load_frames
from src.navngen.frame import Frame # Ensure src.navngen.frame is loaded
from src.navngen.plot import plot_trajectory
from src.navngen.utils import get_subtrajectory, rotation_matrix_to_euler_angles
from src.navngen.load_images import load_image
import matplotlib.pyplot as plt

def print_frame_rel_orientation(frames, start_index, n_frames):
    print(f"Printing orientation for {n_frames} frames starting at index {start_index}:")
    for i in range(start_index, start_index + n_frames):
        if i >= len(frames):
            print(f"Warning: Index {i} is out of bounds for frames list. Stopping.")
            break
        frame = frames[i]
        # Assuming get_essential_matrix() returns R, t and we need R
        R, _ = frame.get_essential_matrix()
        roll, pitch, yaw = rotation_matrix_to_euler_angles(R)
        print(f"Frame {i+1}: Roll={roll:.4f}, Pitch={pitch:.4f}, Yaw={yaw:.4f}")

def print_frame_abs_orientation(frames, start_index, n_frames):
    print(f"Printing orientation for {n_frames} frames starting at index {start_index}:")
    for i in range(start_index, start_index + n_frames):
        if i >= len(frames):
            print(f"Warning: Index {i} is out of bounds for frames list. Stopping.")
            break
        frame = frames[i]
        # Assuming get_essential_matrix() returns R, t and we need R
        R, _ = frame.get_pose()
        roll, pitch, yaw = rotation_matrix_to_euler_angles(R)
        print(f"Frame {i+1}: absolute Roll={roll:.4f}, abs Pitch={pitch:.4f}, abs Yaw={yaw:.4f}")

def print_frame_rel_translation(frames, start_index, n_frames):
    print(f"Printing relative translation for {n_frames} frames starting at index {start_index}:")
    for i in range(start_index, start_index + n_frames):
        if i >= len(frames):
            print(f"Warning: Index {i} is out of bounds for frames list. Stopping.")
            break
        frame = frames[i]
        # Assuming get_essential_matrix() returns R, t_rel
        _, t_rel = frame.get_essential_matrix()
        print(f"Frame {i+1}: Relative Translation: X={t_rel[0]:.4f}, Y={t_rel[1]:.4f}, Z={t_rel[2]:.4f}")

def print_frame_abs_translation(frames, start_index, n_frames):
    print(f"Printing absolute translation for {n_frames} frames starting at index {start_index}:")
    for i in range(start_index, start_index + n_frames):
        if i >= len(frames):
            print(f"Warning: Index {i} is out of bounds for frames list. Stopping.")
            break
        frame = frames[i]
        # Assuming get_pose() returns R, t_abs
        _, t_abs = frame.get_pose()
        print(f"Frame {i+1}: Absolute Translation: X={t_abs[0]:.4f}, Y={t_abs[1]:.4f}, Z={t_abs[2]:.4f}")


if __name__=="__main__":
    # Patch for pickle to find the 'frame' module
    if 'src.navngen.frame' in sys.modules and 'frame' not in sys.modules:
        sys.modules['frame'] = sys.modules['src.navngen.frame']
    
    frame_path = Path(__file__).resolve().parent.parent / "assets" / "outputs" / "01_pickle.pkl.gz"
    
    frames = load_frames(frame_path)

    # Example usage of the new functions
    print_frame_abs_orientation(frames, start_index=440, n_frames=10)
    print_frame_rel_translation(frames, start_index=440, n_frames=10)
    print_frame_abs_translation(frames, start_index=440, n_frames=10)

    focus_frames = get_subtrajectory(frames, 19, 25, 20)

    image0path = Path("/home/joe/data/kitti/color/dataset/sequences/01/image_2/000446.png")
    image1path = Path("/home/joe/data/kitti/color/dataset/sequences/01/image_2/000447.png")

    image0 = load_image(image0path)
    image1 = load_image(image1path)



    frame0 = frames[445]
    frame1 = frames[446]

    path0 = frame0.get_path()
    path1 = frame1.get_path()
    print(f"path 0: {path0}, path1: {path1}")

    matches = frame1.get_matches()

    R0, _ = frame0.get_essential_matrix()
    R1, _ = frame1.get_essential_matrix()

    r0,p0,y0 = rotation_matrix_to_euler_angles(R0)
    r1,p1,y1 = rotation_matrix_to_euler_angles(R1)

    m_kpts0, m_kpts1 = frame0.get_kpts()[matches[..., 0]], frame1.get_kpts()[matches[..., 1]]
    inlier0 = m_kpts0[frame1.get_info()["inliers"]]
    inlier1 = m_kpts1[frame1.get_info()["inliers"]]
    viz2d.plot_images([image0, image1])
    viz2d.plot_matches(inlier0, inlier1, color="lime", lw=0.2)

    plt.show()

