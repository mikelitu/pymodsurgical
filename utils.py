from pathlib import Path, PosixPath
import numpy as np
import torch

def create_save_dir(
    save_dir: PosixPath | str, 
    filename: str,
) -> PosixPath:
    if not isinstance(save_dir, PosixPath):
        save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir()
    return save_dir / filename

def _norm_numpy(
    array: np.ndarray,
    as_img: bool = False
) -> np.ndarray:
    array = (array - array.min()) / (array.max() - array.min())
    if as_img:
        array = (255 * array).astype(np.uint8)
    return array

def _norm_torch(
    tensor: torch.Tensor,
    as_img: bool = False
) -> torch.Tensor:
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    if as_img:
        tensor = (255 * tensor).to(torch.uint8)
    return tensor