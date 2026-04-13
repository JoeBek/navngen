from pathlib import Path
import numpy as np
import yaml
import pandas as pd
from .camera import parse_camera_euroc, parse_camera_kitti, parse_camera_surfnav
from poselib import estimate_relative_pose
from .load_images import load_image
from tqdm import tqdm
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd, map_tensor
from .frame import Frame
from typing import Sequence, Tuple, Optional
import torch
import cv2
from .filter import filter_depth
 
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


def create_frame_sequence(input_path: Path, images_dirname:str) -> Sequence[Frame]:
    """
    Creates a sequence of frames with paths and timestamps from a directory.
    """
    # works for kitti and airport data
    img_path = input_path / images_dirname
    img_files = sorted(img_path.glob('*.png')) + sorted(img_path.glob('*.jpg'))

    with open(input_path / 'times.txt') as f:
        timestamps = [float(line) for line in f]

    # Align images and timestamps
    num_frames = min(len(img_files), len(timestamps))
    img_files = img_files[:num_frames]
    timestamps = timestamps[:num_frames]

    return [Frame(path=p, timestamp=t) for p, t in zip(img_files, timestamps)]


def create_frame_sequence_euroc(mav0_path: Path) -> Sequence[Frame]:
    """
    Creates a sequence of frames from an EuRoC mav0 directory.
    Reads timestamps and filenames from cam0/data.csv.
    Timestamps are in nanoseconds (int).
    """
    cam0_path = mav0_path / 'cam0'
    img_path = cam0_path / 'data'

    df = pd.read_csv(cam0_path / 'data.csv')
    df.rename(columns=lambda x: x.strip(), inplace=True)

    timestamps = df['#timestamp [ns]'].tolist()
    filenames = df['filename'].tolist()

    return [Frame(path=img_path / fn, timestamp=ts) for ts, fn in zip(timestamps, filenames)]


def extract_kpts_from_sequence(frames: Sequence[Frame]) -> Sequence[Frame]:
    """
    Extracts keypoints for each frame in the sequence using frame.path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)

    for i, frame in enumerate(tqdm(frames, desc="Extracting keypoints")):
        if frame.path is None:
            raise ValueError(f"Frame {i} has no path.")
        
        image = load_image(frame.path)
            
        features = extractor.extract(image.to(device))
        
        features_no_batch = rbd(features)
        features_cpu = {k: v.cpu().detach() for k, v in features_no_batch.items()}

        frame.features = features_cpu
        frame.kpts = features_cpu['keypoints']

    return frames

def solve_poses_from_frames(frames: Sequence[Frame], solver: Solver) -> Sequence[Frame]:
    """
    Takes a sequence of partial frames with keypoints, matches them, 
    and solves for the full trajectory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LightGlue(features="superpoint").eval().to(device)

    if not frames:
        return []

    r0 = np.eye(3)
    t0 = np.zeros(3)
    
    frames[0].pose = (r0, t0)
    
    trajectories = {frames[0].timestamp: (r0, t0)}
    
    # The first frame is already "processed" in a sense.
    processed_frames = [frames[0]]

    for i in tqdm(range(1, len(frames)), desc="Solving poses"):
        frame_last = frames[i-1]
        frame_curr = frames[i]

        feats0_cpu = frame_last.features
        feats1_cpu = frame_curr.features

        # Add batch dimension and move to device
        feats0 = map_tensor(feats0_cpu, lambda x: x[None].to(device))
        feats1 = map_tensor(feats1_cpu, lambda x: x[None].to(device))

        matches01 = matcher({"image0": feats0, "image1": feats1})
        matches01 = rbd(matches01)

        kpts0 = frame_last.kpts
        kpts1 = frame_curr.kpts
        matches = matches01['matches'].cpu()

        m_kpts0 = kpts0[matches[..., 0]].numpy()
        m_kpts1 = kpts1[matches[..., 1]].numpy()
        
        transform, info = solver.solve_relative_pose(m_kpts0, m_kpts1)
        r, t = transform.R, transform.t
        
        rl, tl = trajectories[frame_last.timestamp]
        rn, tn = compose_with_unit_direction(rl, tl, r, t)

        trajectories[frame_curr.timestamp] = (rn, tn)

        frame_curr.E = (r, t)
        frame_curr.pose = (rn, tn)
        frame_curr.matches = matches
        frame_curr.info = info
        
        processed_frames.append(frame_curr)

    return processed_frames

