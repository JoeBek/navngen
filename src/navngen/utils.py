
from typing import Sequence
from .frame import Frame


def get_frame_index_by_second(second:int, fps:int) -> int:

    return second * fps

def get_subtrajectory(frames:Sequence[Frame], t_start, t_end, fps) -> Sequence[Frame]:
    
    start_index = get_frame_index_by_second(t_start, fps)
    end_index = get_frame_index_by_second(t_end, fps)
    
    frame_list = list(frames)
    return frame_list[start_index:end_index]
    
        
