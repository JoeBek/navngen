'''
Docstring for run_trajectory_lg

this file is perhaps temporary and is intended to serve as the component that takes a dataloader and runs the algorithms to produce the trajectory

'''
from pathlib import Path
import numpy as np
import yaml
import pandas as pd
from camera import parse_camera_surfnav
from poselib import estimate_relative_pose
from load_images import load_image
from tqdm import tqdm
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from frame import Frame
from typing import Sequence, Tuple
import torch
import cv2
from filter import filter_depth_normalized

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

def parse_camera_kitti(path: Path) -> dict:
    camera_config = {}
    with open(path, 'r') as file:
        for line in file:
            if line.startswith('P0:'):
                parts = line.split()[1:]
                p_matrix = np.array([float(x) for x in parts]).reshape(3,4)
                
                fx = p_matrix[0,0]
                fy = p_matrix[1,1]
                cx = p_matrix[0,2]
                cy = p_matrix[1,2]
                
                camera_config['model'] = 'OPENCV'
                camera_config['params'] = [fx, fy, cx, cy, 0, 0, 0, 0]
                
                return camera_config
                
    raise ValueError("Could not find P0 in calibration file")
        
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



def match_frames(frame0, frame1):
    # load extractor and matcher
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
    matcher = LightGlue(features="superpoint").eval().to(device)


    feats0 = extractor.extract(frame0.to(device))
    feats1 = extractor.extract(frame1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    return  feats1, kpts1.cpu(), matches.cpu(), m_kpts0, m_kpts1

def match_frames_debug(frame0, frame1):
    # load extractor and matcher
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
    matcher = LightGlue(features="superpoint").eval().to(device)


    feats0 = extractor.extract(frame0.to(device))
    feats1 = extractor.extract(frame1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    return kpts0, kpts1, matches 



def load_ground_truth_euroc(path: Path) -> pd.DataFrame:
    #timestamp, p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z, b_w_x, b_w_y, b_w_z, b_a_x, b_a_y, b_a_z
    df = pd.read_csv(path)
    df.rename(columns=lambda x: x.strip(), inplace=True)
    return df

class Solver():
    
    
    
    def __init__(self, config_path:Path, config_type: str = 'surfnav'):
        '''
        expects a yaml file from the surfnav repo
        '''
        
        if config_type == 'surfnav':
            self.camera = parse_camera_surfnav(config_path)
        elif config_type == 'euroc':
            self.camera = parse_camera_euroc(config_path)
        elif config_type == 'kitti':
            self.camera = parse_camera_kitti(config_path)
        else:
            raise ValueError(f"Unknown config_type: {config_type}")


        self.ransac_options = {
            'max_epipolar_error': 0.5,
            'min_inlier_ratio': 0.5,
            'confidence': 0.999,
            'max_iterations': 50000,
            'min_iterations': 1000


        }
       

    def solve_relative_pose(self, m_kpts1, m_kpts2):
        
        pose, info = estimate_relative_pose(m_kpts1, m_kpts2, self.camera, self.camera, self.ransac_options)

        return pose, info
 

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

    for timestamp, filename in tqdm(zip(timestamps, filenames), desc="Storing Image Paths...", total=len(timestamps)):
        frames[timestamp] = img_path / filename

    sorted_timestamps = sorted(frames.keys())

    for i in tqdm(range(1, len(sorted_timestamps)), desc="getting trajectory"):
        ts_last = sorted_timestamps[i-1]
        ts_curr = sorted_timestamps[i]
        
        last_frame_path = frames[ts_last]
        curr_frame_path = frames[ts_curr]

        try:
            frame_last = load_image(last_frame_path)
            frame_curr = load_image(curr_frame_path)
        except Exception as e:
            import traceback, logging
            logging.error(f"operation failed for euroc image loading: %s", e)
            traceback.print_exc()
            exit()


        mk_last, mk_curr = match_frames(frame_last, frame_curr)
        
        transform, info = solver.solve_relative_pose(mk_last.cpu().numpy(), mk_curr.cpu().numpy())
        r = transform.R
        t = transform.t
        
        rl, tl = trajectories[ts_last]

        rn, tn = compose_with_unit_direction(rl, tl, r, t)
        
        trajectories[ts_curr] = (rn, tn)
        
    return trajectories

def random_sample_keypoints(kpts:torch.Tensor, num_kept:int) -> torch.Tensor:
    """
    Randomly samples a subset of keypoints.

    Args:
        kpts: A tensor of keypoints of shape (N, 2).
        num_kept: The number of keypoints to keep.

    Returns:
        A tensor of shape (num_kept, 2) containing the randomly sampled keypoints.
    """
    if num_kept >= kpts.shape[0]:
        return kpts
    indices = torch.randperm(kpts.shape[0])[:num_kept]
    return kpts[indices]

def get_frame_pose_kitti(img_last:torch.Tensor, img_curr:torch.Tensor, num_keypoints:int, solver: Solver) -> Frame:
    """
    Computes the relative pose between two images, with random sampling of keypoints in the second image.

    Args:
        img_last: The previous image as a torch.Tensor.
        img_curr: The current image as a torch.Tensor.
        num_keypoints: The number of keypoints to sample in the current image.
        solver: A Solver object for pose estimation.

    Returns:
        A Frame object containing the computed pose and other information.
    """
    
    feats, kpts_curr, matches, mk_last, mk_curr = match_frames_sampled(img_last, img_curr, num_keypoints)
    
    transform, info = solver.solve_relative_pose(mk_last.cpu().numpy(), mk_curr.cpu().numpy())
    
    r = transform.R
    t = transform.t
    
    output_frame = Frame(
        kpts=kpts_curr.cpu(),
        matches=matches.cpu(),
        essential_matrix = (r, t),
        info=info,
        features=feats
    )
    
    return output_frame

def match_frames_sampled(frame0, frame1, num_keypoints):
    # load extractor and matcher
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
    matcher = LightGlue(features="superpoint").eval().to(device)


    feats0 = extractor.extract(frame0.to(device))
    feats1 = extractor.extract(frame1.to(device))

    # Randomly sample keypoints from feats1
    if num_keypoints < len(feats1['keypoints']):
        indices = torch.randperm(len(feats1['keypoints']))[:num_keypoints]
        feats1['keypoints'] = feats1['keypoints'][indices]
        feats1['descriptors'] = feats1['descriptors'][indices]


    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    return  feats1, kpts1.cpu(), matches.cpu(), m_kpts0, m_kpts1

    
    

def gen_trajectory_kitti(input_path: Path, solver: Solver) -> Sequence[Frame]:
    img_path = input_path / 'image_2'
    
    img_files = sorted(img_path.glob('*.png'))

    output_frames = []

    
    with open(input_path / 'times.txt') as f:
        timestamps = [float(line) for line in f]

    # aachen day 0 has 4469 images but 4470 timestamps. this is a temporary fix
    if len(timestamps) > len(img_files):
        timestamps = timestamps[:len(img_files)]

    first_frame = load_image(img_files[0])
    # in torchscript land, this is W, H, C, but in numpy land its C, H, W
    # in our case we get a torchscript tensor so we must use the appropriate dims
    solver.camera['height'] = first_frame.shape[1]
    solver.camera['width'] = first_frame.shape[2]

    r0 = np.eye(3)
    t0 = np.zeros(3)

    trajectories = {timestamps[0]: (r0, t0)}
    frames = {}

    for timestamp, filename in tqdm(zip(timestamps, img_files), desc="Storing Image Paths...", total=len(timestamps)):
        frames[timestamp] = filename

    sorted_timestamps = sorted(frames.keys())

    for i in tqdm(range(1, len(sorted_timestamps)), desc="getting trajectory"):


        ts_last = sorted_timestamps[i-1]
        ts_curr = sorted_timestamps[i]

        
        last_frame_path = frames[ts_last]
        curr_frame_path = frames[ts_curr]


        try:
            frame_last = load_image(last_frame_path)
            frame_curr = load_image(curr_frame_path)
        except Exception as e:
            import traceback, logging
            logging.error(f"operation failed for kitti image loading: %s", e)
            traceback.print_exc()
            exit()


        
        feats , kpts_curr, matches, mk_last, mk_curr = match_frames(frame_last, frame_curr)
    
        
        transform, info = solver.solve_relative_pose(mk_last.cpu().numpy(), mk_curr.cpu().numpy())
        r = transform.R
        t = transform.t
        
        rl, tl = trajectories[ts_last]

        rn, tn = compose_with_unit_direction(rl, tl, r, t)

        output_frame = Frame(
            kpts=kpts_curr.cpu(),
            matches=matches.cpu(),
            path=curr_frame_path,
            essential_matrix=(r, t),
            pose=(rn, tn),
            features=feats,
            timestamp=ts_curr,
            info=info
        )
        output_frames.append(output_frame)
        

        
        trajectories[ts_curr] = (rn, tn)
        
    return output_frames 


def gen_trajectory_kitti_depth_filtered(input_path: Path, solver: Solver, depth_path: Path, tl: float = 0.0, th: float = 0.0) -> Sequence[Frame]:
    img_path = input_path / 'image_2'
    
    img_files = sorted(img_path.glob('*.png'))
    depth_files = sorted(depth_path.glob('*.png'))

    output_frames = []

    
    with open(input_path / 'times.txt') as f:
        timestamps = [float(line) for line in f]

    # aachen day 0 has 4469 images but 4470 timestamps. this is a temporary fix
    if len(timestamps) > len(img_files):
        timestamps = timestamps[:len(img_files)]

    if len(img_files) != len(depth_files):
        raise ValueError(f"Number of images ({len(img_files)}) does not match number of depth maps ({len(depth_files)})")

    first_frame = load_image(img_files[0])
    # in torchscript land, this is W, H, C, but in numpy land its C, H, W
    # in our case we get a torchscript tensor so we must use the appropriate dims
    solver.camera['height'] = first_frame.shape[1]
    solver.camera['width'] = first_frame.shape[2]

    r0 = np.eye(3)
    t0 = np.zeros(3)

    trajectories = {timestamps[0]: (r0, t0)}
    frames = {}
    depth_map_paths = {}

    for timestamp, filename, d_filename in tqdm(zip(timestamps, img_files, depth_files), desc="Storing Image Paths...", total=len(timestamps)):
        frames[timestamp] = filename
        depth_map_paths[timestamp] = d_filename

    sorted_timestamps = sorted(frames.keys())

    for i in tqdm(range(1, len(sorted_timestamps)), desc="getting trajectory"):

        ts_last = sorted_timestamps[i-1]
        ts_curr = sorted_timestamps[i]

        
        last_frame_path = frames[ts_last]
        curr_frame_path = frames[ts_curr]
        curr_depth_path = depth_map_paths[ts_curr]


        try:
            frame_last = load_image(last_frame_path)
            frame_curr = load_image(curr_frame_path)
            depth_curr = cv2.imread(str(curr_depth_path), cv2.IMREAD_UNCHANGED)
        except Exception as e:
            import traceback, logging
            logging.error(f"operation failed for kitti image loading: %s", e)
            traceback.print_exc()
            exit()


        
        feats , kpts_curr, matches, mk_last, mk_curr = match_frames(frame_last, frame_curr)
        
        # Depth filtering for pose estimation
        if depth_curr is not None:
            mk_curr_np = mk_curr.cpu().numpy()
            x = np.round(mk_curr_np[:, 0]).astype(int)
            y = np.round(mk_curr_np[:, 1]).astype(int)
            
            h, w = depth_curr.shape[:2]
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)
            
            matched_kpt_depth = depth_curr[y, x].astype(np.float32)
            
            # Apply thresholds to matches
            mask = matched_kpt_depth >= tl
            if th > tl:
                mask &= (matched_kpt_depth <= th)
            
            mk_last = mk_last[mask]
            mk_curr = mk_curr[mask]
            
            # Sample depth for ALL keypoints in the current frame for the Frame object
            kpts_curr_np = kpts_curr.cpu().numpy()
            xk = np.round(kpts_curr_np[:, 0]).astype(int)
            yk = np.round(kpts_curr_np[:, 1]).astype(int)
            xk = np.clip(xk, 0, w - 1)
            yk = np.clip(yk, 0, h - 1)
            all_kpt_depth = torch.from_numpy(depth_curr[yk, xk].astype(np.float32))
        else:
            all_kpt_depth = None

        
        transform, info = solver.solve_relative_pose(mk_last.cpu().numpy(), mk_curr.cpu().numpy())
        r = transform.R
        t = transform.t
        
        rl, tl_last = trajectories[ts_last]

        rn, tn = compose_with_unit_direction(rl, tl_last, r, t)

        output_frame = Frame(
            kpts=kpts_curr,
            matches=matches,
            path=curr_frame_path,
            essential_matrix=(r, t),
            pose=(rn, tn),
            features=feats,
            timestamp=ts_curr,
            info=info,
            kpt_depth=all_kpt_depth
        )
        
        # Apply depth filtering to the Frame object itself
        if output_frame.kpt_depth is not None:
            filter_depth_normalized([output_frame], tl=tl, th=th)
            
        output_frames.append(output_frame)
        
        trajectories[ts_curr] = (rn, tn)
        
    return output_frames 


def run_euroc_test():

    root = Path(__file__).resolve().parent.parent.parent / "assets" / "V2_01_easy" / "mav0"  
    input_path =  root / "data"
    config_path = root / "cam0" / "sensor.yaml"

    
    write_path = Path(__file__).resolve().parent.parent.parent / "assets" / "outputs" / "v201easy_tum.txt"
    
    solver = Solver(config_path, config_type="euroc")
    traj = gen_trajectory_euroc(root, solver)

    from export_trajectory import convert_euroc_to_tum, export_trajectory_tum
    
    tum_traj = convert_euroc_to_tum(traj)
    export_trajectory_tum(tum_traj, write_path)
 

def run_kitti_test():
    root = Path(__file__).resolve().parent.parent.parent / "assets" / "sequences" / "01"
    config_path = root / "calib.txt"
    depth_path = Path("/home/joe/vt/research/glue/depth/output/01_depth/")

    
    write_path = Path(__file__).resolve().parent.parent.parent / "assets" / "outputs" / "01_tum_depth_0-50.txt"
    pickle_path = Path(__file__).resolve().parent.parent.parent / "assets" / "outputs" / "01_pickle_depth_0-50.pkl.gz"
    
    solver = Solver(config_path, config_type="kitti")
    traj = gen_trajectory_kitti_depth_filtered(root, solver,depth_path, tl=0,th=155)

    from export_trajectory import convert_tum, export_trajectory_tum, export_frames
    
    tum_traj = convert_tum(traj)
    export_trajectory_tum(tum_traj, write_path)
    export_frames(traj,pickle_path)
 
def kitti_test_subsets(num_trials: int, keypoint_amount: int, img_index: int):
    """
    Runs multiple trials of 'get_frame_pose_kitti' on a specific image pair
    from the KITTI dataset and exports the resulting frames.

    Args:
        num_trials: The number of times to run get_frame_pose_kitti for the image pair.
        keypoint_amount: The number of keypoints to sample for each trial.
        img_index: The index of the current image in the KITTI sequence (0-based).
                   The function will use this image and the one prior to it.
    """
    root = Path(__file__).resolve().parent.parent.parent / "assets" / "sequences" / "01"
    config_path = root / "calib.txt"
    
    output_dir = Path(__file__).resolve().parent.parent.parent / "assets" / "outputs" / f"kp_trials_{keypoint_amount}_idx_{img_index}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    solver = Solver(config_path, config_type="kitti")

    from export_trajectory import export_frames
    
    img_path_root = root / 'image_2'
    img_files = sorted(img_path_root.glob('*.png'))

    if not (1 <= img_index < len(img_files)):
        raise ValueError(f"img_index ({img_index}) must be within the valid range [1, {len(img_files) - 1}] for image pairs.")

    frame_last_img = load_image(img_files[img_index - 1])
    frame_curr_img = load_image(img_files[img_index])

    trial_frames: Sequence[Frame] = []

    # Read timestamps for the specific images
    with open(root / 'times.txt') as f:
        timestamps = [float(line) for line in f]
    if len(timestamps) > len(img_files):
        timestamps = timestamps[:len(img_files)]

    for i in tqdm(range(num_trials), desc=f"Running {num_trials} trials for {keypoint_amount} kpts on image pair {img_index-1}-{img_index}"):
        
        current_frame = get_frame_pose_kitti(frame_last_img, frame_curr_img, keypoint_amount, solver)
        
        current_frame.path = img_files[img_index]
        current_frame.timestamp = timestamps[img_index]

        trial_frames.append(current_frame)
    
    pickle_path = output_dir / f"result_frames.pkl.gz"
    export_frames(trial_frames, pickle_path)

    print(f"Saved {num_trials} trial frames for keypoint_amount={keypoint_amount} and img_index={img_index} to {pickle_path}")


if __name__ == "__main__":
    

    kitti_test_subsets(num_trials=100, keypoint_amount=50, img_index=10)