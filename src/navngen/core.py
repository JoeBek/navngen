from vo_base import VO
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

@dataclass
class State:
    curr: int = 1
    trajectories: List[Tuple[float]] = field(default_factory=list)
    


class Navngen():
    

    def __init__(self, vo:VO, frames):
        self.vo = vo
        self.state = State()
        self.frames = frames

    def step(self):
        '''
        perform one VO step. This involves acquiring the next point for the trajectory list
        '''
        
        last = self.state.curr - 1
        
        frame_last = self.frames[last]
        frame_curr = self.frames[self.state.curr]
        
        feats_last = self.vo.extract(frame_last)
        feats_curr = self.vo.extract(frame_curr)
        
        matches = self.vo.match(feats_last, feats_curr)
        
        relative_pose = self.vo.get_pose(matches, feats_last, feats_curr)
        
        # unit direction vector
        
    
    


    