from pathlib import Path
from typing import Sequence
import numpy as np
import torch
import cv2
from typing import Callable, List, Optional, Tuple, Union, Any
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from frame import Frame
'''
load images from storage into memory. use PIL to read from pathlib object input and output as np array or cv mat or tensor array
'''


def get_paths(root:Path) -> List[Path]:
    paths = [p for p in root.iterdir() if p.is_file()]
    try:
        paths.sort(key=lambda p: int(p.stem))
    except ValueError:
        paths.sort(key=lambda p: p.name)

    return paths



def resize_image(
    image: np.ndarray,
    size: Union[List[int], int],
    fn: str = "max",
    interp: Optional[str] = "area",
) -> Tuple[np.ndarray, Any]:
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]

    fnc = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fnc(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    modes = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }
    mode = cv2.INTER_LINEAR if not interp else modes[interp]

    return cv2.resize(image, dsize=(w_new, h_new), interpolation=mode), scale
    


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image

def load_image(path: Path, resize: Optional[int] = None, **kwargs) -> torch.Tensor:
    image = read_image(path)
    if resize is not None:
        image, _ = resize_image(image, resize, **kwargs)
    return numpy_image_to_torch(image)
    

def load_depth_mono(frames: Sequence[Frame], depth_path: Path) -> Sequence[Frame]:
    """
    loads depth information into frames. This is done by querying the depth image at keypoints and adding the information to the frame objects.

    :param frames: Sequence of Frame objects
    :param depth_path: Path to the directory containing depth images
    :return: The updated sequence of Frame objects
    """
    depth_paths = get_paths(depth_path)
    
    if len(depth_paths) != len(frames):
        raise ValueError(f"Number of depth images ({len(depth_paths)}) does not match number of frames ({len(frames)})")

    for frame, d_path in zip(frames, depth_paths):
        # Load depth image. Using IMREAD_UNCHANGED to preserve depth precision (e.g. 16-bit)
        # If the user says it's normalized, it might be 8-bit or float.
        depth_img = cv2.imread(str(d_path), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise IOError(f"Could not read depth image at {d_path}")
        
        if frame.kpts is not None:
            kpts = frame.kpts
            if torch.is_tensor(kpts):
                kpts_numpy = kpts.cpu().numpy()
            else:
                kpts_numpy = np.array(kpts)

            # Assuming kpts are (N, 2) with (x, y) coordinates
            # We need to sample depth_img[y, x]
            # Rounding to nearest pixel for indexing
            x = np.round(kpts_numpy[:, 0]).astype(int)
            y = np.round(kpts_numpy[:, 1]).astype(int)

            # Clip coordinates to be within image bounds
            h, w = depth_img.shape[:2]
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)

            # Sample depth
            kpt_depth = depth_img[y, x]
            
            # Convert back to torch tensor if frame uses tensors
            frame.kpt_depth = torch.from_numpy(kpt_depth.astype(np.float32))
            
    return frames

class TrajectoryDataset(Dataset):
    

    def __init__(self, root:Path):
        '''
        Docstring for __init__
        
        :param self: Description
        :param path: Description
        :type path: Path
        
        loads images from path into memory
        '''
        self.root = root
        self.paths = get_paths(root)
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index) -> Any:
        '''
        Docstring for __getitem__
        
        :param self: Description
        :param index: Description
        :return: Description
        :rtype: Any ... should be a torch.Tensor


        '''
        path = self.paths[index]
        return load_image(path)

    
if __name__ == "__main__":
    
    this_file = Path(__file__).resolve()
    
    v2_01_easy = Path("V2_01_easy") / "mav0" / "cam0" / "data"
    root = this_file.parent.parent.parent / "assets" / v2_01_easy
    print(f"root path: {root}")
    
    dataset = TrajectoryDataset(root)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=64,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
    )

    for batch_idx, images in tqdm(enumerate(dataloader), desc="batch number..."):
        
        for image in tqdm(images, desc="processing images"):
            
            
            print("---")
            print("Small image kernel: ")
            for i in range(3):
                print(f"{image[0,0,i]} {image[0,1,i]} {image[0,2,i]}")
            print("---")


    

