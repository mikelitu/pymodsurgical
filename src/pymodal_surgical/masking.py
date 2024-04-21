import cv2
import numpy as np
from pathlib import PosixPath, Path
import torch
import torch.nn.functional as F
from pymodal_surgical.video_reader import VideoType


class Masking(object):
    def __init__(
        self, 
        mask_path: PosixPath | str,
        video_type: VideoType = VideoType.MONO
    ) -> None:
        if isinstance(mask_path, str):
            mask_path = Path(mask_path)

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found at {mask_path}")
        
        self.mask = self._load_mask(mask_path)
        # Normalize mask to be between 0 and 1
        self.mask = (self.mask / 255).astype(np.float32)
        self.width = self.mask.shape[1]
        self.height = self.mask.shape[0]
        
        if video_type == VideoType.STEREO:
            self.left_mask, self.right_mask = self._split_mask()
        
        self.video_type = video_type
    
    def _split_mask(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mask[..., :self.width // 2], self.mask[..., self.width // 2:]
    
    def _check_stereo(self, camera_pos: str | None = None) -> None:
        if camera_pos is None:
            raise ValueError("Camera position must be specified for stereo video")
        if camera_pos == "left":
            self.mask = self.left_mask
        elif camera_pos == "right":
            self.mask = self.right_mask
        else:
            raise ValueError("Camera position must be either 'left' or 'right'")

    def apply_mask(self, frames: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return frames * self.mask[None, None, ...]
    
    def __call__(self, frames: np.ndarray | torch.Tensor, camera_pos: str | None = None) -> np.ndarray | torch.Tensor:
        
        if self.video_type == VideoType.STEREO:
            self._check_stereo(camera_pos)
        
        if isinstance(frames, torch.Tensor):
            self.mask = torch.from_numpy(self.mask).to(frames.device) if not isinstance(self.mask, torch.Tensor) else self.mask
            if self.width != frames.shape[3] or self.height != frames.shape[2]:
                self.mask = F.interpolate(self.mask[None, None, ...], (frames.shape[2], frames.shape[3]), mode="nearest")[0, 0]
        
        else:
            if self.width != frames[0].shape[1] or self.height != frames[0].shape[0]:
                self.mask = cv2.resize(self.mask, (frames[0].shape[2], frames[0].shape[1]), interpolation=cv2.INTER_NEAREST)
        
        masked_frames = self.apply_mask(frames)
        return masked_frames
    
    @property
    def masking(self) -> np.ndarray:
        if self.video_type == VideoType.STEREO:
            return self.left_mask
        else:
            return self.mask

    @staticmethod
    def _load_mask(mask_path: PosixPath | str) -> np.ndarray:
        if isinstance(mask_path, PosixPath):
            mask_path = str(mask_path)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return mask

