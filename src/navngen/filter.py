
from frame import Frame
from typing import Sequence
import torch

def filter_depth_normalized(frames: Sequence[Frame], tl=0.0, th=0.0) -> Sequence[Frame]:
    """
    Filters out keypoints based on depth thresholds. 
    Returns frames with an equal or reduced number of keypoints based on normalized depth thresholds.

    :param frames: A sequence of Frame objects to be filtered.
    :type frames: Sequence[Frame]
    :param tl: The lower depth threshold. Keypoints with depth below this value will be removed.
    :type tl: float
    :param th: The upper depth threshold. If greater than tl, keypoints with depth above this value will be removed.
    :type th: float
    :return: The sequence of filtered Frame objects.
    :rtype: Sequence[Frame]
    """
    for frame in frames:
        if frame.kpt_depth is None or frame.kpts is None or frame.features is None:
            continue

        # Ensure kpt_depth is a tensor
        if not isinstance(frame.kpt_depth, torch.Tensor):
            frame.kpt_depth = torch.tensor(frame.kpt_depth)

        # Create a boolean mask for keypoints within the depth range
        mask = frame.kpt_depth >= tl
        if th > tl:
            mask &= (frame.kpt_depth <= th)

        # Apply the mask to keypoints and their associated data
        frame.kpts = frame.kpts[mask]
        frame.kpt_depth = frame.kpt_depth[mask]

        # The 'features' dictionary from SuperPoint/LightGlue needs to be filtered as well
        # to maintain consistency.
        if 'keypoints' in frame.features:
            frame.features['keypoints'] = frame.features['keypoints'][mask]
        if 'keypoint_scores' in frame.features:
            frame.features['keypoint_scores'] = frame.features['keypoint_scores'][mask]
        if 'descriptors' in frame.features:
            # Descriptors have a different shape (N, D), so we filter on the first dimension
            frame.features['descriptors'] = frame.features['descriptors'][mask]
            
        # Note: This function does not update frame.matches, as it would require
        # re-indexing across multiple frames, which is outside the typical scope
        # of a simple keypoint filter. Matches should be recomputed if needed.

    return frames
