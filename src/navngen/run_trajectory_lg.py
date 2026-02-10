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
from typing import Sequence
import torch

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
            'max_error': 1.5,
            'min_inlier_ratio': 0.5,
            'confidence': 0.999,
            'max_iterations': 1000
        }
       

    def solve_relative_pose(self, m_kpts1, m_kpts2):
        
        pose, info = estimate_relative_pose(m_kpts1, m_kpts2, self.camera, self.camera, self.ransac_options)

        return pose
 

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
        
        transform = solver.solve_relative_pose(mk_last.cpu().numpy(), mk_curr.cpu().numpy())
        r = transform.R
        t = transform.t
        
        rl, tl = trajectories[ts_last]

        rn, tn = compose_with_unit_direction(rl, tl, r, t)
        
        trajectories[ts_curr] = (rn, tn)
        
    return trajectories


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
    
        
        transform = solver.solve_relative_pose(mk_last.cpu().numpy(), mk_curr.cpu().numpy())
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
            timestamp=ts_curr
        )
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

    
    write_path = Path(__file__).resolve().parent.parent.parent / "assets" / "outputs" / "01_tum.txt"
    pickle_path = Path(__file__).resolve().parent.parent.parent / "assets" / "outputs" / "01_pickle.pkl.gz"
    
    solver = Solver(config_path, config_type="kitti")
    traj = gen_trajectory_kitti(root, solver)

    from export_trajectory import convert_tum, export_trajectory_tum, export_frames
    
    tum_traj = convert_tum(traj)
    export_trajectory_tum(tum_traj, write_path)
    export_frames(traj,pickle_path)
 


if __name__ == "__main__":
    

    run_kitti_test()