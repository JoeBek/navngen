import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from pathlib import Path
from lightglue import SuperPoint 
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d
import torch
import matplotlib.pyplot as plt
from typing import Union, Any
import cv2
import base64
import io
import yaml

def fig_to_numpy(fig, close=True):
    """Return RGB uint8 (H, W, 3) from a Matplotlib figure."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = rgba[..., :3].copy()
    if close:
        plt.close(fig)
    return rgb


def apply_sp(frame: torch.Tensor, debug=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
        
    feats0 = extractor.extract(frame.to(device))
    kpts0 = feats0["keypoints"]

    viz2d.plot_images([frame])
    viz2d.plot_keypoints(kpts0, ps=6, colors="red")

    if debug:
        plt.show()
        return
    # get plot as numpy
   
    fig = plt.gcf()
    kp_frame = fig_to_numpy(fig)
    return kp_frame,  feats0


def create_dirs(path:Path):
    
    output_path = path / "outputs"
    if not output_path.exists() or not output_path.is_dir():
        print(f"creating dir: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
            
    images_path = output_path / "images"
    if not images_path.exists() or not images_path.is_dir():
        print(f"creating dir: {images_path}")
        images_path.mkdir(parents=True, exist_ok=True)

    features_path = output_path / "features"
    if not features_path.exists() or not features_path.is_dir():
        print(f"creating dir: {features_path}")
        features_path.mkdir(parents=True, exist_ok=True)

    
    return (images_path, features_path)

def parse_input_path():
    """
    Parse command-line -i / --input path and return a pathlib.Path (exists check).
    Usage: script.py -i /path/to/file_or_dir
    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Parse input path")
    parser.add_argument(
        "-i", "--input", required=True, type=Path, help="Input file or directory path"
    )
    args = parser.parse_args()
    input_path: Path = args.input

    if not input_path.exists():
        parser.error(f"input path does not exist: {input_path}")

    return input_path



from PIL import Image

def save_image(arr, path, format=None, quality=95, normalize=False):
    """
    Save a numpy array to PNG or JPEG based on path or explicit format.
    - arr: ndarray, shape (H,W), (H,W,3) or (H,W,4)
    - path: str or Path
    - format: "PNG" or "JPEG" (optional — inferred from suffix)
    - quality: JPEG quality 1-100
    - normalize: if True and arr is float, scale to 0..1 before uint8 conversion
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    a = np.asarray(arr)
    if a.ndim == 2:
        mode = "L"
    elif a.ndim == 3 and a.shape[2] == 3:
        mode = "RGB"
    elif a.ndim == 3 and a.shape[2] == 4:
        mode = "RGBA"
    else:
        raise ValueError(f"Unsupported array shape: {a.shape}")

    # convert floats to uint8
    if np.issubdtype(a.dtype, np.floating):
        if normalize:
            mn, mx = float(a.min()), float(a.max())
            if mx > mn:
                a = (a - mn) / (mx - mn)
            else:
                a = np.clip(a, 0.0, 1.0)
        a = (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    else:
        a = np.clip(a, 0, 255).astype(np.uint8)

    # decide format
    ext = (format or p.suffix.lstrip(".")).lower()
    fmt = "JPEG" if ext in ("jpg", "jpeg") else "PNG"

    img = Image.fromarray(a, mode=mode)

    # JPEG does not support alpha; composite over white or convert
    if fmt == "JPEG":
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])  # alpha composite over white
            img = bg
        else:
            img = img.convert("RGB")
        img.save(str(p), format=fmt, quality=int(quality), optimize=True)
    else:
        img.save(str(p), format=fmt, optimize=True)

def _ensure_numpy(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def save_features(path, keypoints, descriptors, keypoint_scores, image_size=None, fmt="npz"):
    """
    Save keypoint data to disk.
    - path: str or Path. For fmt=="npz" extension is respected; for fmt=="yaml" a .yml will be written.
    - keypoints: (N,2)
    - descriptors: (N,D)
    - keypoint_scores: (N,)
    - image_size: optional metadata
    fmt: "npz" (default, compact/binary) or "yaml" (base64-encoded numpy binaries inside YAML)
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    kp = _ensure_numpy(keypoints)
    desc = _ensure_numpy(descriptors)
    ks = _ensure_numpy(keypoint_scores)
    imgsz = _ensure_numpy(image_size)

    if fmt.lower() == "npz":
        # compact binary format, recommended
        np.savez_compressed(
            str(p),
            keypoints=kp,
            descriptors=desc,
            keypoint_scores=ks,
            image_size=imgsz,
        )
        return

    # yaml fallback: store each array as base64-encoded numpy .npy bytes
    def _enc(arr):
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    data = {
        "keypoints": _enc(kp),
        "descriptors": _enc(desc),
        "keypoint_scores": _enc(ks),
        "image_size": _enc(imgsz) if imgsz is not None else None,
        "meta": {
            "format": "npy_base64",
        },
    }
    with open(str(p), "w") as f:
        yaml.safe_dump(data, f)



def decode_features(feats):
    
    keypoints = feats["keypoints"].cpu().numpy()
    keypoint_scores = feats["keypoint_scores"].cpu().numpy()
    descriptors = feats["descriptors"].cpu().numpy()
    image_size = feats["image_size"].cpu().numpy()

    return keypoints, keypoint_scores, descriptors, image_size


if __name__ == "__main__":
    

    
    cwd = Path.cwd()
    
    images_path, features_path = create_dirs(cwd)

    input_path = parse_input_path() 
    
    for path in tqdm(input_path.glob("*"), desc="Writing Images..."):
        
        # assume each path is an image
        frame = None
        try:
            frame = load_image(path)
        except Exception as e:
            import traceback, logging
            logging.error("operation failed: %s", e)
            traceback.print_exc()
            exit()
        # frame loaded
        
        new_frame, feats = apply_sp(frame)

        keypoints, keypoint_scores, descriptors, image_size = decode_features(feats)

        
        # save features
        feature_path = features_path / (path.stem + "_features" + ".npz")       
        save_features(feature_path,keypoints, keypoint_scores, descriptors, image_size, fmt="npz")
        

        # save image
        frame_path = images_path / (path.stem + "_output" + path.suffix)
        save_image(new_frame,frame_path, "jpg")
        
        
        

    
    