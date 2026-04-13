
from .frame import Frame
from typing import Sequence
import torch
import numpy as np

def filter_depth(frames: Sequence[Frame], normalize=False, tl=0.0, th=0.0) -> Sequence[Frame]:
    """
    Filters out keypoints based on absolute depth thresholds.
    Returns frames with an equal or reduced number of keypoints based on depth values.

    :param frames: A sequence of Frame objects to be filtered.
    :type frames: Sequence[Frame]
    :param tl: The lower depth threshold. Keypoints with depth below this value will be removed.
    :type tl: float
    :param th: The upper depth threshold. If greater than tl, keypoints with depth above this value will be removed.
    :type th: float
    :return: The sequence of filtered Frame objects.
    :rtype: Sequence[Frame]
    """

    if normalize:
        frames = normalize_depth(frames)
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
        if frame.kpt_depth is not None:
            frame.kpt_depth = frame.kpt_depth[mask]

        # Filter features to maintain consistency
        if 'keypoints' in frame.features:
            frame.features['keypoints'] = frame.features['keypoints'][mask]
        if 'keypoint_scores' in frame.features:
            frame.features['keypoint_scores'] = frame.features['keypoint_scores'][mask]
        if 'descriptors' in frame.features:
            frame.features['descriptors'] = frame.features['descriptors'][mask]
            
    return frames


def filter_segmentation(frames: Sequence[Frame], masks, filter_ids: Sequence[int]) -> Sequence[Frame]:
    """
    Filters out keypoints based on segmentation masks.
    Keeps keypoints that fall into specified mask classes.

    :param frames: A sequence of Frame objects to be filtered.
    :type frames: Sequence[Frame]
    :param masks: An iterable of segmentation masks (H, W), one for each frame.
                  May be a list or a lazy generator to reduce peak RAM usage.
    :param filter_ids: A sequence of class IDs to keep.
    :type filter_ids: Sequence[int]
    :return: The sequence of filtered Frame objects.
    :rtype: Sequence[Frame]
    """
    if hasattr(masks, '__len__') and len(frames) != len(masks):
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


def normalize_depth(frames: Sequence[Frame]) -> Sequence[Frame]:
    """
    Aggregates all depth masks and normalizes depth points to a relative distance
    from 0 to 1, where 0 is the closest.

    :param frames: A sequence of Frame objects to be processed.
    :type frames: Sequence[Frame]
    :return: The sequence of Frame objects with normalized depth.
    :rtype: Sequence[Frame]
    """
    all_depths = []
    for frame in frames:
        if frame.kpt_depth is not None:
            if isinstance(frame.kpt_depth, torch.Tensor):
                all_depths.append(frame.kpt_depth)
            else:
                all_depths.append(torch.tensor(frame.kpt_depth))

    if not all_depths:
        return frames

    all_depths_tensor = torch.cat(all_depths)
    if all_depths_tensor.numel() == 0:
        return frames

    min_depth = torch.min(all_depths_tensor)
    max_depth = torch.max(all_depths_tensor)

    if max_depth > min_depth:
        # Normalize between 0 and 1
        scale = max_depth - min_depth
        for frame in frames:
            if frame.kpt_depth is not None:
                # Ensure kpt_depth is a tensor for the operation
                if not isinstance(frame.kpt_depth, torch.Tensor):
                    frame.kpt_depth = torch.tensor(frame.kpt_depth, dtype=torch.float32)
                
                frame.kpt_depth = (frame.kpt_depth - min_depth) / scale
    else:
        # If all depths are the same, they are all the 'closest', so map to 0
        for frame in frames:
            if frame.kpt_depth is not None:
                frame.kpt_depth = torch.zeros_like(frame.kpt_depth)
                
    return frames
