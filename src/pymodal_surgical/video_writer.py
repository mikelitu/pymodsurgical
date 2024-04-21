import cv2
import numpy as np
from pathlib import Path
import torch
from video_reader import VideoType
from torchvision.utils import flow_to_image


class VideoWriter(object):

    def __init__(
        self,
        video_path: str | Path,
        video_config: dict[str, str | int | float | bool | VideoType],
    ) -> None:
        
        if not isinstance(video_path, Path):
            video_path = Path(video_path)

        self.video_path = video_path
        self.video_config = video_config
        self.video_type = video_config["video_type"]
        self._writing_method = self._write_mono if self.video_type == VideoType.MONO else self._write_stereo

    
    def _write_mono(self, frames: np.ndarray | torch.Tensor) -> None:
        if isinstance(frames, torch.Tensor):
            _, height, width = frames[0].shape
        else:
            height, width, _ = frames[0].shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(self.video_path), fourcc, self.video_config["fps"], (width, height))
        
        for frame in frames:
            
            if frame.shape[0] == 2 and isinstance(frame, torch.Tensor):
                frame = flow_to_image(frame)
                frame = frame.permute(1, 2, 0).cpu().numpy()
                
            # Ensure frame is uint8
            if frame.dtype == torch.float32:
                frame = (frame * 255).to(dtype=torch.uint8).cpu().numpy()
            
            elif frame.dtype == np.float32:
                frame = (frame * 255).astype(np.uint8)

            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()

    
    def _write_stereo(self, frames: tuple[list[np.ndarray], list[np.ndarray]]) -> None:
        height, width, _ = frames[0][0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(self.video_path), fourcc, self.video_config["fps"], (width * 2, height))
        for i in range(len(frames[0])):
            frame = np.concatenate((cv2.cvtColor(frames[0][i], cv2.COLOR_RGB2BGR), cv2.cvtColor(frames[1][i], cv2.COLOR_RGB2BGR)), axis=1)
            out.write(frame)
        out.release()

    def __call__(self, frames: np.ndarray | torch.Tensor) -> None:
        self._writing_method(frames)
    