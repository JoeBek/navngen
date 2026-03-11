
from .frame import Frame
from typing import Sequence
import torch
import numpy as np

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

def filter_segmentation(frames: Sequence[Frame], masks: Sequence[np.ndarray], filter_ids: Sequence[int]) -> Sequence[Frame]:
    """
    Filters out keypoints based on segmentation masks.
    Keeps keypoints that fall into specified mask classes.

    :param frames: A sequence of Frame objects to be filtered.
    :type frames: Sequence[Frame]
    :param masks: A sequence of segmentation masks (H, W), one for each frame.
    :type masks: Sequence[np.ndarray]
    :param filter_ids: A sequence of class IDs to keep.
    :type filter_ids: Sequence[int]
    :return: The sequence of filtered Frame objects.
    :rtype: Sequence[Frame]
    """
    if len(frames) != len(masks):
        raise ValueError("Number of frames and masks must be equal.")

    for frame, mask_img in zip(frames, masks):
        if frame.kpts is None or frame.features is None:
            continue

        kpts_np = frame.kpts.cpu().numpy()
        # Keypoints are (x, y), need to convert to integer indices for the mask
        # We'll use floor to be safe, as keypoints can be sub-pixel.
        kpt_indices = np.floor(kpts_np).astype(int)

        # Ensure indices are within mask bounds
        height, width = mask_img.shape
        x_indices, y_indices = kpt_indices[:, 0], kpt_indices[:, 1]
        
        valid_indices_mask = (x_indices >= 0) & (x_indices < width) & \
                             (y_indices >= 0) & (y_indices < height)
        
        # Start with a mask of all False
        segmentation_mask = np.zeros(len(frame.kpts), dtype=bool)

        # Get the class IDs for the valid keypoints from the segmentation mask
        # Note: In image coordinates, the first index is y (rows) and the second is x (columns)
        valid_kpt_classes = mask_img[y_indices[valid_indices_mask], x_indices[valid_indices_mask]]
        
        # Create a boolean mask for keypoints that are in the desired classes
        # This checks if each item in `valid_kpt_classes` is in `filter_ids`
        in_filter_mask = np.isin(valid_kpt_classes, filter_ids)

        # Update the full segmentation mask at the positions of the valid keypoints
        segmentation_mask[valid_indices_mask] = in_filter_mask
        
        # Convert boolean numpy array to a torch tensor for masking torch tensors
        torch_mask = torch.from_numpy(segmentation_mask).to(frame.kpts.device)

        # Apply the mask
        frame.kpts = frame.kpts[torch_mask]
        if frame.kpt_depth is not None:
            frame.kpt_depth = frame.kpt_depth[torch_mask]

        if 'keypoints' in frame.features:
            frame.features['keypoints'] = frame.features['keypoints'][torch_mask]
        if 'keypoint_scores' in frame.features:
            frame.features['keypoint_scores'] = frame.features['keypoint_scores'][torch_mask]
        if 'descriptors' in frame.features:
            frame.features['descriptors'] = frame.features['descriptors'][torch_mask]
            
    return frames
