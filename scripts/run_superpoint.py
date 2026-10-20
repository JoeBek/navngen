import sys
import argparse
from pathlib import Path

import numpy as np
import torch

# Add the project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from lightglue import SuperPoint
from lightglue.utils import rbd
from src.navngen.load_images import load_image
from tqdm import tqdm

IMAGE_GLOBS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")


def find_images(input_path: Path, recursive: bool) -> list[Path]:
    """Return a sorted list of image files under input_path (or the file itself)."""
    if input_path.is_file():
        return [input_path]
    globber = input_path.rglob if recursive else input_path.glob
    files: set[Path] = set()
    for pattern in IMAGE_GLOBS:
        files.update(globber(pattern))
    return sorted(files)


def main(args):
    """
    Run SuperPoint over every image in a directory and persist the features to
    disk as one compressed .npz per image.
    """
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    extractor = SuperPoint(max_num_keypoints=args.max_num_keypoints).eval().to(device)

    images = find_images(args.input_path, args.recursive)
    if not images:
        print(f"No images found under {args.input_path}")
        return

    args.output_path.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(images, desc="Running SuperPoint"):
        image = load_image(img_path)
        features = rbd(extractor.extract(image.to(device)))
        features_np = {k: v.cpu().numpy() for k, v in features.items()}

        out_file = args.output_path / f"{img_path.stem}.npz"
        np.savez_compressed(out_file, image_path=str(img_path), **features_np)

    print(f"Saved SuperPoint features for {len(images)} images to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SuperPoint over a directory of images and persist the "
        "keypoints/descriptors as one compressed .npz per image.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input_path", type=Path, required=True,
                        help="Directory containing images (or a single image file).")
    parser.add_argument("--output_path", type=Path, required=True,
                        help="Directory to write the per-image .npz feature files.")
    parser.add_argument("--max_num_keypoints", type=int, default=2048,
                        help="Maximum number of keypoints per image.")
    parser.add_argument("--recursive", action="store_true",
                        help="Recurse into subdirectories when searching for images.")
    parser.add_argument("--device", type=str, default="",
                        help="Torch device, e.g. 'cuda' or 'cpu'. Defaults to cuda if available.")

    args = parser.parse_args()
    main(args)
