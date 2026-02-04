import yaml
from pathlib import Path
from poselib import estimate_relative_pose
from match import match_frames
from tqdm import tqdm
from batch_kp import apply_sp, decode_features
from lightglue.utils import load_image
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
from scipy.spatial.transform import Rotation


def parse_camera_surfnav(path:Path) -> dict:
    
    input_config = {}
    camera_config = {}
    with open(path, 'r') as file:
        input_config = yaml.safe_load(file)
        
    camera_config['model'] = 'OPENCV'
    
    width = 0
    height = 0.30
    cam_matrix = []
    dist_matrix = []
    try:
        width = input_config['image_width']
        height = input_config['image_height']
        cam_matrix = input_config['camera_matrix']['data']
        dist_matrix = input_config['distortion_coefficients']['data']
    
    except KeyError as e:
        print(f'Error trying to parse surfnav data: {e}')
        
    assert(len(cam_matrix) == 9)
    
    # extract intrinsics and distortion coefficients in the format poselib expects
    camera_data = [cam_matrix[0], cam_matrix[2], cam_matrix[4], cam_matrix[5]]
    camera_data.extend(dist_matrix)

    camera_config['width'] = width
    camera_config['height'] = height
    camera_config['params'] = camera_data 
    
    return camera_config
    



class Solver():
    
    
    
    def __init__(self, config_path:Path, config_type: str = 'surfnav'):
        '''
        expects a yaml file from the surfnav repo
        '''
        
        if config_type == 'surfnav':
            self.camera = parse_camera_surfnav(config_path)
        elif config_type == 'euroc':
            self.camera = parse_camera_euroc(config_path)
        else:
            raise ValueError(f"Unknown config_type: {config_type}")


        self.ransac_options = {
            'max_error': 1.5,
            'min_inlier_ratio': 0.5,
            'confidence': 0.999,
            'max_iterations': 1000
        }
       

    def solve_relative_pose(self, m_kpts1, m_kpts2):
        
        pose, info = estimate_relative_pose(m_kpts1, m_kpts2, self.camera, self.camera, self.ransac_options)

        return pose
        
def pose_to_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
        
def compose_with_unit_direction(R_prev: np.ndarray, t_prev: np.ndarray,
                                R_rel: np.ndarray, u_rel: np.ndarray,
                                scale: float = 1.0):
    """
    Apply relative rotation R_rel and unit translation direction u_rel
    (expressed in the previous camera frame) with scalar scale.
    Returns (R_new, t_new).
    """
    # ensure u_rel is a 3-vector
    u = np.asarray(u_rel, dtype=float).reshape(3)
    if np.linalg.norm(u) > 0:
        t_rel = u / np.linalg.norm(u) * float(scale)
    else:
        t_rel = np.zeros(3, dtype=float)

    R_new = R_prev @ R_rel
    t_new = t_prev + R_prev @ t_rel
    return R_new, t_new

def gen_trajectory(input_path, solver:Solver):
    """
    generate a set of trajectories from a path 
    """
    
    r0 = np.eye(3)
    t0 = np.zeros(3)
    

    trajectories = [(r0, t0)]
    frames = []

    # sort pathnames
    paths = [p for p in input_path.iterdir() if p.is_file()]
    try:
        paths.sort(key=lambda p: int(p.stem))
    except ValueError:
        paths.sort(key=lambda p: p.name)

    # iterate over paths
    for path in tqdm(paths, desc="Loading Images..."):
        

    
    # assume each path is an image
        try:
            frame = load_image(path)
            frames.append(frame)
        except Exception as e:
            import traceback, logging
            logging.error("operation failed: %s", e)
            traceback.print_exc()
            exit()
        # frame loaded
    
    for i in tqdm(range(1,len(frames)), desc="getting trajectory"):
        
        frame_last = frames[i-1]
        frame_curr = frames[i]

        mk_last, mk_curr= match_frames(frame_last,frame_curr)
        
        transform = solver.solve_relative_pose(mk_last.cpu().numpy(), mk_curr.cpu().numpy())
        r = transform.R
        t = transform.t
        
        rl, tl = trajectories[len(trajectories) - 1]

        rn, tn = compose_with_unit_direction(rl,tl,r,t)
        
        trajectories.append((rn,tn))
        
    return trajectories

        

def plot_trajectory(trajectories: list[tuple[np.ndarray, np.ndarray]],
                    plane: str = "xz",
                    ax=None,
                    show: bool = True):
    """
    Plot bird's-eye view of trajectory translations.

    - trajectories: list of (R, t) tuples where t is a 3-vector.
    - plane: which 2D plane to plot:
        'xz' (default) -> lateral (x) vs forward (z) top-down view,
        'xy' -> x vs y,
        'yz' -> y vs z.
    - ax: optional matplotlib Axes to draw on.
    - show: if True call plt.show().

    Returns the Axes used.
    """
    # extract translations and ensure numpy array with shape (N,3)
    ts = np.array([np.asarray(t, dtype=float).reshape(3,) for (_, t) in trajectories])
    if ts.ndim != 2 or ts.shape[1] < 2:
        raise ValueError("translations must be 3-vectors")

    if plane == "xz":
        xs, ys = ts[:, 0], ts[:, 2]
        xlabel, ylabel = "x", "z"
    elif plane == "xy":
        xs, ys = ts[:, 0], ts[:, 1]
        xlabel, ylabel = "x", "y"
    elif plane == "yz":
        xs, ys = ts[:, 1], ts[:, 2]
        xlabel, ylabel = "y", "z"
    else:
        raise ValueError("plane must be one of 'xz','xy','yz'")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True

    # Create a color gradient from green (start) to red (end)
    n = len(xs)
    if n < 1:
        return ax

    # t goes from 0 (start, green) to 1 (end, red)
    t = np.linspace(0.0, 1.0, n)
    colors = np.vstack((t, 1.0 - t, np.zeros_like(t))).T  # RGB: (r, g, 0)

    # Use LineCollection to draw colored segments between points
    points = np.column_stack((xs, ys))
    if len(points) > 1:
        segments = np.stack([points[:-1], points[1:]], axis=1)
        seg_colors = colors[:-1]
        lc = LineCollection(segments, colors=seg_colors, linewidths=1.5)
        ax.add_collection(lc)
    else:
        # Single point, just scatter it
        ax.scatter(xs, ys, c=[colors[0]], s=20)

    # Scatter individual points with the same gradient
    ax.scatter(xs, ys, c=colors, s=20, edgecolors='face')

    # Mark start and end explicitly (larger markers)
    ax.scatter(xs[0], ys[0], c=[colors[0]], s=60, label="start", edgecolors='k')
    ax.scatter(xs[-1], ys[-1], c=[colors[-1]], s=60, label="end", edgecolors='k')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.set_title("Visual Odometry on Airport Data")
    ax.legend()

    # autoscale to include the added LineCollection
    ax.relim()
    ax.autoscale_view()

    if show and created_fig:
        plt.show()

    return ax

def parse_camera_euroc(path: Path) -> dict:
    camera_config = {}
    with open(path, 'r') as file:
        input_config = yaml.safe_load(file)

    camera_config['model'] = 'OPENCV'
    
    #verified from https://github.com/poselib/poselib/blob/main/python/poselib.cc#L292C25-L292C25
    #fx, fy, cx, cy
    camera_data = input_config['intrinsics']
    camera_data.extend(input_config['distortion_coefficients'])

    camera_config['width'] = input_config['resolution'][0]
    camera_config['height'] = input_config['resolution'][1]
    camera_config['params'] = camera_data
    
    return camera_config

def load_ground_truth_euroc(path: Path) -> pd.DataFrame:
    #timestamp, p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z, b_w_x, b_w_y, b_w_z, b_a_x, b_a_y, b_a_z
    df = pd.read_csv(path)
    df.rename(columns=lambda x: x.strip(), inplace=True)
    return df

def gen_trajectory_euroc(input_path: Path, solver: Solver):
    cam0_path = input_path / 'cam0'
    img_path = cam0_path / 'data'
    
    df = pd.read_csv(cam0_path / 'data.csv')
    df.rename(columns=lambda x: x.strip(), inplace=True)
    
    timestamps = df['#timestamp [ns]']
    filenames = df['filename']
    
    r0 = np.eye(3)
    t0 = np.zeros(3)
    
    trajectories = {timestamps[0]: (r0, t0)}
    frames = {}

    for timestamp, filename in tqdm(zip(timestamps, filenames), desc="Loading Images...", total=len(timestamps)):
        try:
            frame = load_image(img_path / filename)
            frames[timestamp] = frame
        except Exception as e:
            import traceback, logging
            logging.error("operation failed: %s", e)
            traceback.print_exc()
            exit()

    sorted_timestamps = sorted(frames.keys())

    for i in tqdm(range(1, len(sorted_timestamps)), desc="getting trajectory"):
        ts_last = sorted_timestamps[i-1]
        ts_curr = sorted_timestamps[i]
        
        frame_last = frames[ts_last]
        frame_curr = frames[ts_curr]

        mk_last, mk_curr = match_frames(frame_last, frame_curr)
        
        transform = solver.solve_relative_pose(mk_last.cpu().numpy(), mk_curr.cpu().numpy())
        r = transform.R
        t = transform.t
        
        rl, tl = trajectories[ts_last]

        rn, tn = compose_with_unit_direction(rl, tl, r, t)
        
        trajectories[ts_curr] = (rn, tn)
        
    return trajectories

def align_and_plot_trajectory(vo_traj, gt_df):
    
    gt_timestamps = gt_df['#timestamp'].values
    gt_positions = gt_df[['p_x', 'p_y', 'p_z']].values
    gt_orientations = gt_df[['q_w', 'q_x', 'q_y', 'q_z']].values

    vo_timestamps = np.array(list(vo_traj.keys()))
    vo_positions = np.array([t for R, t in vo_traj.values()])

    # Synchronize trajectories
    common_timestamps = np.intersect1d(vo_timestamps, gt_timestamps)
    
    vo_indices = [np.where(vo_timestamps == ts)[0][0] for ts in common_timestamps]
    gt_indices = [np.where(gt_timestamps == ts)[0][0] for ts in common_timestamps]

    vo_positions_synced = vo_positions[vo_indices]
    gt_positions_synced = gt_positions[gt_indices]

    # Align trajectories (using Horn's method for similarity transform)
    vo_mean = np.mean(vo_positions_synced, axis=0)
    gt_mean = np.mean(gt_positions_synced, axis=0)

    vo_centered = vo_positions_synced - vo_mean
    gt_centered = gt_positions_synced - gt_mean

    M = vo_centered.T @ gt_centered
    
    U, S, Vt = np.linalg.svd(M)
    
    R_align = (Vt.T @ U.T)

    if np.linalg.det(R_align) < 0:
        U[:, -1] *= -1
        R_align = (Vt.T @ U.T)

    scale = np.sum(S) / np.sum(np.diag(vo_centered.T @ vo_centered))
    
    t_align = gt_mean - scale * R_align @ vo_mean

    # Apply alignment to the whole VO trajectory
    vo_positions_aligned = scale * (R_align @ vo_positions.T).T + t_align
    
    #compute ATE
    errors = gt_positions_synced - vo_positions_synced
    ate = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
    
    fig, ax = plt.subplots()

    # Plot ground truth trajectory
    ax.plot(gt_positions[:, 0], gt_positions[:, 2], label='Ground Truth', color='b')
    
    # Plot aligned VO trajectory
    ax.plot(vo_positions_aligned[:, 0], vo_positions_aligned[:, 2], label='VO Trajectory (Aligned)', color='r')

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.set_title(f"Visual Odometry vs. Ground Truth (ATE: {ate:.4f})")
    ax.legend()
    plt.show()
    

if __name__ == "__main__":

    #
    # OLD CODE
    #
    # path = Path("configs/camera_forward.yaml")
    # solver = Solver(path)
    # traj_path = Path("assets/lot_nav_images")

    # trajectories = gen_trajectory(traj_path, solver)
    # plot_trajectory(trajectories)
    
    
    # exit()
    # path1 = Path("assets/output0349.png")
    # path2 = Path("assets/output0350.png")

    # m_kpts1, m_kpts2 = match_images(path1, path2)
    
    # print(len(m_kpts1))
    # print(len(m_kpts2))

    # pose = solver.solve_relative_pose(m_kpts1, m_kpts2)
    
    #
    # END OF OLD CODE
    #
    
    # TODO: Change this path to the root of the EuRoC MAV dataset
    euroc_path = Path("assets/V2_01_easy/mav0")
    
    config_path = euroc_path / 'cam0' / 'sensor.yaml'
    solver = Solver(config_path, config_type='euroc')
    
    gt_path = euroc_path / 'state_groundtruth_estimate0' / 'data.csv'
    gt_df = load_ground_truth_euroc(gt_path)
    
    vo_traj = gen_trajectory_euroc(euroc_path, solver)
    
    align_and_plot_trajectory(vo_traj, gt_df)

    
    print("goodbye, world")