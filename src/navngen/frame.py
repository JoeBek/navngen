
from typing import Sequence, Optional, Tuple, TypeAlias
import torch
from pathlib import Path
import numpy as np

# ( R  t ) as numpy arrays. Might change to Tensor for consistency
PoseType: TypeAlias = Tuple[np.ndarray, np.ndarray]

class Frame:
    features: Optional[dict]
    kpts: Optional[torch.Tensor]
    matches: Optional[torch.Tensor]
    path: Optional[Path]
    E: Optional[PoseType]
    pose: Optional[PoseType]
    timestamp: Optional[int]
    info: Optional[dict]
    '''
    Docstring for Frame

    class for debugging frames
    contains all information you might want
    image index (pathname), sp output, lg output, estimated essential matrix, estimated pose
    

    '''
    
    
    def __init__(self, features:Optional[dict]=None, kpts:Optional[torch.Tensor]=None,
                  matches:Optional[torch.Tensor]=None, path:Optional[Path]=None, 
                  essential_matrix:Optional[PoseType]=None, pose:Optional[PoseType]=None, timestamp:Optional[int]=None, info:Optional[dict]=None, kpt_depth:Optional[torch.Tensor]=None):

        self.features = features
        self.kpts=kpts
        self.matches=matches
        self.path=path
        self.E = essential_matrix
        self.pose = pose
        self.timestamp =timestamp 
        self.info = info
        self.kpt_depth = kpt_depth

    def get_kpt_depth(self) -> torch.Tensor:
        if self.kpt_depth is None:
            raise RuntimeError("info not set for this Frame.")
        return self.kpt_depth



    def get_info(self) -> dict:
        if self.info is None:
            raise RuntimeError("info not set for this Frame.")
        return self.info


    def get_features(self) -> dict:
        if self.features is None:
            raise RuntimeError("Features not set for this Frame.")
        return self.features

    def get_kpts(self) -> torch.Tensor:
        if self.kpts is None:
            raise RuntimeError("Keypoints not set for this Frame.")
        return self.kpts

    def get_matches(self) -> torch.Tensor:
        if self.matches is None:
            raise RuntimeError("Matches not set for this Frame.")
        return self.matches

    def get_path(self) -> Path:
        if self.path is None:
            raise RuntimeError("Path not set for this Frame.")
        return self.path

    def get_essential_matrix(self) -> PoseType:
        if self.E is None:
            raise RuntimeError("Essential matrix not set for this Frame.")
        return self.E

    def get_pose(self) -> PoseType:
        if self.pose is None:
            raise RuntimeError("Pose not set for this Frame.")
        return self.pose
    
    def get_timestamp(self) -> int:
        if self.timestamp is None:
            raise RuntimeError("Timestamp not set for this Frame.")
        return self.timestamp

    def set_kpt_depth(self, kpt_depth):
        self.kpt_depth = kpt_depth

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp

    def set_features(self, features):
        self.features = features

    def set_kpts(self, kpts):
        self.kpts = kpts

    def set_matches(self, matches):
        self.matches = matches

    def set_path(self, path):
        self.path = path

    def set_essential_matrix(self, essential_matrix):
        self.E = essential_matrix

    def set_pose(self, pose):
        self.pose = pose


        