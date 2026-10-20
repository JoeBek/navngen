import sys
import argparse
from pathlib import Path

import numpy as np
import torch

# Add the project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd, map_tensor
from src.navngen.load_images import load_image
from tqdm import tqdm

IMAGE_GLOBS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
FEATURE_KEYS = ("keypoints", "keypoint_scores", "descriptors", "image_size")


def find_images(input_path: Path, recursive: bool) -> list[Path]:
    globber = input_path.rglob if recursive else input_path.glob
    files: set[Path] = set()
    for pattern in IMAGE_GLOBS:
        files.update(globber(pattern))
    return sorted(files)


def features_from_images(input_path: Path, recursive: bool, max_kpts: int,
                         device: torch.device) -> list[tuple[str, dict]]:
    """Extract SuperPoint features on the fly from a directory of images."""
    images = find_images(input_path, recursive)
    if not images:
        return []
    extractor = SuperPoint(max_num_keypoints=max_kpts).eval().to(device)
    out: list[tuple[str, dict]] = []
    for p in tqdm(images, desc="Extracting SuperPoint"):
        feats = rbd(extractor.extract(load_image(p).to(device)))
        out.append((p.stem, {k: v.cpu() for k, v in feats.items()}))
    return out


def features_from_npz(features_path: Path) -> list[tuple[str, dict]]:
    """Load precomputed SuperPoint features written by run_superpoint.py."""
    files = sorted(features_path.glob("*.npz"))
    out: list[tuple[str, dict]] = []
    for p in files:
        data = np.load(p, allow_pickle=True)
        feats = {k: torch.from_numpy(data[k]) for k in FEATURE_KEYS if k in data.files}
        out.append((p.stem, feats))
    return out


def main(args):
    """
    Match consecutive frames with LightGlue and persist the matches to disk as
    one compressed .npz per image pair.
    """
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    if args.features_path:
        frames = features_from_npz(args.features_path)
    else:
        frames = features_from_images(args.input_path, args.recursive,
                                      args.max_num_keypoints, device)

    if len(frames) < 2:
        print(f"Need at least 2 frames to match; found {len(frames)}.")
        return

    matcher = LightGlue(features="superpoint").eval().to(device)
    args.output_path.mkdir(parents=True, exist_ok=True)

    n_pairs = 0
    with torch.inference_mode():
        for (name0, f0), (name1, f1) in tqdm(
            list(zip(frames[:-1], frames[1:])), desc="Matching with LightGlue"
        ):
            fb0 = map_tensor(f0, lambda x: x[None].to(device))
            fb1 = map_tensor(f1, lambda x: x[None].to(device))

            matches01 = rbd(matcher({"image0": fb0, "image1": fb1}))

            matches = matches01["matches"].cpu().numpy()
            scores = matches01["scores"].cpu().numpy()

            out_file = args.output_path / f"{name0}__{name1}.npz"
            np.savez_compressed(
                out_file,
                name0=name0,
                name1=name1,
                matches=matches,
                scores=scores,
                keypoints0=f0["keypoints"].numpy(),
                keypoints1=f1["keypoints"].numpy(),
            )
            n_pairs += 1

    print(f"Saved LightGlue matches for {n_pairs} pairs to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Match consecutive frames with LightGlue and persist the "
        "matches as one compressed .npz per image pair.\n"
        "Provide either --features_path (output of run_superpoint.py) or "
        "--input_path (images; features are extracted on the fly).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--features_path", type=Path,
                        help="Directory of SuperPoint .npz files from run_superpoint.py.")
    source.add_argument("--input_path", type=Path,
                        help="Directory of images; SuperPoint features are extracted on the fly.")

    parser.add_argument("--output_path", type=Path, required=True,
                        help="Directory to write the per-pair match .npz files.")
    parser.add_argument("--max_num_keypoints", type=int, default=2048,
                        help="Max keypoints per image (only used with --input_path).")
    parser.add_argument("--recursive", action="store_true",
                        help="Recurse into subdirectories (only used with --input_path).")
    parser.add_argument("--device", type=str, default="",
                        help="Torch device, e.g. 'cuda' or 'cpu'. Defaults to cuda if available.")

    args = parser.parse_args()
    main(args)
