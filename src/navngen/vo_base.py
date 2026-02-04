from abc import ABC, abstractmethod
from typing import Tuple


class VO(ABC):
    
    @abstractmethod
    def extract(self, frame):
        """return Features"""
    
    @abstractmethod
    def match(self, frame1, frame2):
        """Return area"""
    
    @abstractmethod
    def get_pose(self, matches, features1, features2) -> Tuple:
        """Return pose"""