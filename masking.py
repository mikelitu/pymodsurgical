import cv2
import numpy as np
from pathlib import PosixPath
import torch

class Masking(object):
    def __init__(self, mask_path: PosixPath | str) -> None:
        self.mask = self._load_mask(mask_path)
        # Normalize mask to be between 0 and 1
        self.mask = (self.mask / 255).astype(np.float32)

    def apply_mask(self, frame: np.ndarray) -> list[np.ndarray]:
        return frame * self.mask[..., None]
    
    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        if isinstance(frames, torch.Tensor):
            self.mask = torch.from_numpy(self.mask).to(frames.device)
        
        return [self.apply_mask(frame) for frame in frames]

    @staticmethod
    def _load_mask(mask_path: PosixPath | str) -> np.ndarray:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return mask

