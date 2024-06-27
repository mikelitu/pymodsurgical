import cv2
import numpy as np
from pathlib import PosixPath, Path
import torch
import torch.nn.functional as F
from .reader import VideoType


class Masking(object):
    """
    A class for applying masks to video frames.

    Args:
        mask_path (PosixPath | Path | str): The path to the mask file.
        video_type (VideoType, optional): The type of video. Defaults to VideoType.MONO.

    Attributes:
        mask (np.ndarray): The loaded mask.
        width (int): The width of the mask.
        height (int): The height of the mask.
        left_mask (np.ndarray): The left half of the mask (for stereo videos).
        right_mask (np.ndarray): The right half of the mask (for stereo videos).
        video_type (VideoType): The type of video.

    """

    def __init__(
        self, 
        mask_path: PosixPath | Path | str,
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
        """
        Split the mask into left and right halves (for stereo videos).

        Returns:
            tuple[np.ndarray, np.ndarray]: The left and right halves of the mask.

        """
        return self.mask[..., :self.width // 2], self.mask[..., self.width // 2:]
    
    def _check_stereo(self, camera_pos: str | None = None) -> None:
        """
        Check if the video is stereo and set the mask accordingly.

        Args:
            camera_pos (str | None, optional): The camera position. Defaults to None.

        Raises:
            ValueError: If camera position is not specified for stereo video.
            ValueError: If camera position is not 'left' or 'right'.

        """
        if camera_pos is None:
            raise ValueError("Camera position must be specified for stereo video")
        if camera_pos == "left":
            self.mask = self.left_mask
        elif camera_pos == "right":
            self.mask = self.right_mask
        else:
            raise ValueError("Camera position must be either 'left' or 'right'")

    def apply_mask(self, frames: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Apply the mask to the frames.

        Args:
            frames (np.ndarray | torch.Tensor): The frames to apply the mask to.

        Returns:
            np.ndarray | torch.Tensor: The masked frames.

        """
        return frames * self.mask[None, None, ...]
    
    def __call__(self, frames: np.ndarray | torch.Tensor, camera_pos: str | None = None) -> np.ndarray | torch.Tensor:
        """
        Apply the mask to the frames.

        Args:
            frames (np.ndarray | torch.Tensor): The frames to apply the mask to.
            camera_pos (str | None, optional): The camera position. Defaults to None.

        Returns:
            np.ndarray | torch.Tensor: The masked frames.

        """
        if self.video_type == VideoType.STEREO:
            self._check_stereo(camera_pos)
        
        if self.video_type == VideoType.STEREO:
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
        """
        Get the mask.

        Returns:
            np.ndarray: The mask.

        """
        if self.video_type == VideoType.STEREO:
            return self.left_mask
        else:
            return self.mask

    @staticmethod
    def _load_mask(mask_path: PosixPath | str) -> np.ndarray:
        """
        Load the mask from the given path.

        Args:
            mask_path (PosixPath | str): The path to the mask file.

        Returns:
            np.ndarray: The loaded mask.

        """
        if isinstance(mask_path, PosixPath) or isinstance(mask_path, Path):
            mask_path = str(mask_path)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return mask

