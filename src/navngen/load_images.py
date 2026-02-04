from pathlib import Path
from typing import Sequence
import numpy as np
import torch
import cv2
from typing import Callable, List, Optional, Tuple, Union, Any
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
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


    

