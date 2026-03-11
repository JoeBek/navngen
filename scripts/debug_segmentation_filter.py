
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
from src.navngen.filter import filter_segmentation
from lightglue import SuperPoint, viz2d
from lightglue.utils import rbd

def main(args):
    """
    Loads an image, extracts keypoints, filters them based on a segmentation mask,
    and visualizes the result.
    """
    if 'src.navngen.frame' in sys.modules and 'frame' not in sys.modules:
        sys.modules['frame'] = sys.modules['src.navngen.frame']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image and segmentation mask
    image = load_image(args.image_path, grayscale=False) # load as color for visualization
    image_gray = load_image(args.image_path, grayscale=True)
    mask = np.load(args.mask_path)

    # Extract keypoints
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    features = extractor.extract(image_gray.to(device))
    features = rbd(features) # remove batch dimension

    # Create a Frame object
    frame = Frame(
        path=args.image_path,
        kpts=features['keypoints'],
        features=features
    )

    # Store original keypoints for visualization
    original_kpts = frame.kpts.clone().cpu()

    # Filter keypoints using the segmentation mask
    frames_to_filter = [frame]
    masks_to_filter = [mask]
    filter_ids = [int(fid) for fid in args.filter_ids.split(',')]

    filtered_frames = filter_segmentation(frames_to_filter, masks_to_filter, filter_ids)

    filtered_frame = filtered_frames[0]
    filtered_kpts = filtered_frame.kpts.cpu()

    print(f"Original keypoints: {len(original_kpts)}")
    print(f"Filtered keypoints: {len(filtered_kpts)}")

    # Visualize the results
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), dpi=100)
    
    # Plot original keypoints
    axes[0].set_title("Original Keypoints")
    viz2d.plot_images([image.cpu()])
    viz2d.plot_keypoints([original_kpts], ps=6)

    # Plot filtered keypoints
    axes[1].set_title(f"Filtered Keypoints (classes: {args.filter_ids})")
    viz2d.plot_images([image.cpu()])
    viz2d.plot_keypoints([filtered_kpts], ps=6)

    for ax in axes:
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug segmentation filter for keypoints.")
    parser.add_argument("--image_path", type=Path, required=True, help="Path to the input image.")
    parser.add_argument("--mask_path", type=Path, required=True, help="Path to the segmentation mask (.npy file).")
    parser.add_argument("--filter_ids", type=str, required=True, 
                        help="Comma-separated list of class IDs to keep. "
                             "Cityscapes class mappings:\n"
                             "  0: road, 1: sidewalk, 2: building, 3: wall, 4: fence,\n"
                             "  5: pole, 6: traffic light, 7: traffic sign, 8: vegetation,\n"
                             "  9: terrain, 10: sky, 11: person, 12: rider, 13: car,\n"
                             "  14: truck, 15: bus, 16: train, 17: motorcycle, 18: bicycle.")

    args = parser.parse_args()
    main(args)
