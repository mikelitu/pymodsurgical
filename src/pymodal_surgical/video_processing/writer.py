import cv2
import numpy as np
from pathlib import Path
import torch
from .reader import VideoType
from torchvision.utils import flow_to_image
from ..utils import create_save_dir


class VideoWriter(object):
    """
    A class for writing video files.

    Args:
        video_config (dict): A dictionary containing video configuration parameters.
            - video_path (str): The path to the output video file.
            - video_type (VideoType): The type of video (MONO or STEREO).
            - fps (int): The frames per second of the video.

    Attributes:
        video_path (Path): The path to the output video file.
        video_config (dict): The video configuration parameters.
        video_type (VideoType): The type of video (MONO or STEREO).
        _writing_method (function): The method used for writing the video frames.

    Methods:
        __call__(frames): Writes the video frames using the selected writing method.

    """

    def __init__(
        self,
        video_config: dict[str, str | int | float | bool | VideoType],
    ) -> None:
        """
        Initializes a VideoWriter object.

        Args:
            video_config (dict): A dictionary containing video configuration parameters.
                - video_path (str): The path to the output video file.
                - video_type (VideoType): The type of video (MONO or STEREO).
                - fps (int): The frames per second of the video.

        """
        video_path = video_config["video_path"]
        if not isinstance(video_path, Path):
            video_path = Path(video_path)

        video_path = create_save_dir(video_path.parent, video_path.name)

        self.video_path = video_path
        self.video_config = video_config
        self.video_type = video_config["video_type"]
        self._writing_method = self._write_mono if self.video_type == VideoType.MONO else self._write_stereo

    def _write_mono(self, frames: np.ndarray | torch.Tensor) -> None:
        """
        Writes mono video frames to the output video file.

        Args:
            frames (np.ndarray or torch.Tensor): The video frames to be written.

        """
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
        """
        Writes stereo video frames to the output video file.

        Args:
            frames (tuple): A tuple containing two lists of video frames (left and right).

        """
        height, width, _ = frames[0][0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(self.video_path), fourcc, self.video_config["fps"], (width * 2, height))
        for i in range(len(frames[0])):
            frame = np.concatenate((cv2.cvtColor(frames[0][i], cv2.COLOR_RGB2BGR), cv2.cvtColor(frames[1][i], cv2.COLOR_RGB2BGR)), axis=1)
            out.write(frame)

        out.release()

    def __call__(self, frames: np.ndarray | torch.Tensor) -> None:
        """
        Writes the video frames using the selected writing method.

        Args:
            frames (np.ndarray or torch.Tensor): The video frames to be written.

        """
        self._writing_method(frames)
    