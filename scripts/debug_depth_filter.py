
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.navngen.load_images import load_image
from src.navngen.frame import Frame
from src.navngen.filter import filter_depth, normalize_depth
from lightglue import SuperPoint, viz2d
from lightglue.utils import rbd

def get_kpt_depth(kpts: torch.Tensor, depth_map: np.ndarray) -> torch.Tensor:
    """Samples depth for each keypoint."""
    # kpts are (x, y), so we need to convert to integer indices.
    kpt_indices = torch.floor(kpts).long()
    
    # Ensure indices are within depth map bounds
    height, width = depth_map.shape
    x_indices, y_indices = kpt_indices[:, 0], kpt_indices[:, 1]
    
    valid_mask = (x_indices >= 0) & (x_indices < width) & (y_indices >= 0) & (y_indices < height)
    
    # Initialize depths with a value indicating invalidity (e.g., -1 or NaN)
    # Using -1 as a placeholder for out-of-bounds keypoints.
    depths = torch.full((len(kpts),), -1.0, dtype=torch.float32)

    # Get depths for valid keypoints.
    # In numpy, indexing is (row, col) which corresponds to (y, x).
    valid_depths = depth_map[y_indices[valid_mask], x_indices[valid_mask]]
    
    depths[valid_mask] = torch.from_numpy(valid_depths.astype(np.float32))
    
    return depths

def main(args):
    """
    Loads an image and depth map, extracts keypoints, filters them based on depth,
    and visualizes the result.
    """
    if 'src.navngen.frame' in sys.modules and 'frame' not in sys.modules:
        sys.modules['frame'] = sys.modules['src.navngen.frame']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image and depth map
    image = load_image(args.image_path, grayscale=False) # load as color for visualization
    image_gray = load_image(args.image_path, grayscale=True)
    depth_map = np.load(args.depth_path)

    # Extract keypoints
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    features = extractor.extract(image_gray.to(device))
    features = rbd(features) # remove batch dimension

    kpts = features['keypoints']
    kpt_depth = get_kpt_depth(kpts.cpu(), depth_map)

    # Create a Frame object
    frame = Frame(
        path=args.image_path,
        kpts=kpts,
        features=features,
        kpt_depth=kpt_depth.to(device) # Move depth to device
    )

    # Store original keypoints for visualization
    original_kpts = frame.kpts.clone().cpu()

    # Filter keypoints using the depth map
    frames_to_filter = [frame]
    
    filtered_frames = filter_depth(frames_to_filter, normalize=args.normalize, tl=args.tl, th=args.th)


    filtered_frame = filtered_frames[0]
    filtered_kpts = filtered_frame.kpts.cpu()

    print(f"Original keypoints: {len(original_kpts)}")
    print(f"Filtered keypoints: {len(filtered_kpts)}")

    # Visualize the results
    # Plot original keypoints
    viz2d.plot_images([image.cpu()])
    viz2d.plot_keypoints([original_kpts], ps=6)

    # Plot filtered keypoints
    viz2d.plot_images([image.cpu()])
    viz2d.plot_keypoints([filtered_kpts], ps=6)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug depth filter for keypoints.")
    parser.add_argument("--image_path", type=Path, required=False, help="Path to the input image.")
    parser.add_argument("--depth_path", type=Path, required=True, help="Path to the depth map (.npy file).")
    parser.add_argument("--tl", type=float, default=0.0, required=False, help="Lower depth threshold.")
    parser.add_argument("--th", type=float, default=0.0, required=False, help="Upper depth threshold.")
    parser.add_argument("--stats", action="store_true", help="Print depth mask statistics (mean, median) of the raw depth map.")
    parser.add_argument("--normalize", action="store_true", help="Normalize the depth points after filtering.")

    args = parser.parse_args()

    if args.stats:
        # Calculate and print statistics of the depth map
        depth_map = np.load(args.depth_path)
        print("\n--- Depth Map Statistics ---")
        print(f"Mean: {np.mean(depth_map):.4f}")
        print(f"Median: {np.median(depth_map):.4f}")
        print("----------------------------\n")
        # Exit after printing stats if only stats were requested.
        # This assumes the user only wants stats and not the full debugging visualization.
        # If the user wants both, this can be adjusted.
        sys.exit(0)
    
    main(args)

