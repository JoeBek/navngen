
from typing import Sequence

class Frame:
    '''
    Docstring for Frame

    class for debugging frames
    contains all information you might want
    image index (pathname), sp output, lg output, estimated essential matrix, estimated pose
    

    '''

    
    def __init__(self, features=None, kpts=None, matches=None, path=None, essential_matrix=None, pose=None):

        self.features = features
        self.kpts=kpts
        self.matches=matches
        self.path=path
        self.E = essential_matrix
        self.pose = pose

    def get_features(self):
        return self.features

    def get_kpts(self):
        return self.kpts

    def get_matches(self):
        return self.matches

    def get_path(self):
        return self.path

    def get_essential_matrix(self):
        return self.E

    def get_pose(self):
        return self.pose

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


def get_frame_index_by_second(second:int, fps:int) -> int:

    return second * fps

    
    
        
        