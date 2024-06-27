import cv2
import numpy as np
from pathlib import Path, PosixPath
from enum import Enum
import torch

from enum import Enum

class VideoType(str, Enum):
    """
    Enum class representing the type of video.

    Attributes:
        STEREO (str): Represents a stereo video.
        MONO (str): Represents a monocular video.
    """
    STEREO = "stereo"
    MONO = "mono"

from enum import Enum

class RetType(str, Enum):
    """
    Enumeration class representing the return type options for video processing.

    Attributes:
        NUMPY (str): Option for returning video frames as numpy arrays.
        TENSOR (str): Option for returning video frames as tensors.
        LIST (str): Option for returning video frames as a list.
    """
    NUMPY = "numpy"
    TENSOR = "tensor"
    LIST = "list"


class VideoReader(object):
    """
    A class for reading video files and extracting frames.

    Args:
        video_config (dict[str, VideoType | str | PosixPath]): A dictionary containing video configuration parameters.
        return_type (RetType | str, optional): The desired return type of the frames. Defaults to RetType.NUMPY.

    Attributes:
        cap (cv2.VideoCapture): The OpenCV video capture object.
        video_path (PosixPath | str): The path to the video file.
        video_config (dict[str, VideoType | str | PosixPath]): The video configuration parameters.
        video_length (int): The total number of frames in the video.
        video_type (VideoType): The type of the video (MONO or STEREO).
        width (int): The width of the video frames.
        height (int): The height of the video frames.
        fps (float): The frames per second of the video.
        _reading_method (function): The method used for reading frames from the video.
        _frame_reading_method (function): The method used for reading a single frame from the video.
        _return_func (function): The function used for returning the frames in the desired format.
        _left_calibration_matrix (np.ndarray): The calibration matrix for the left camera (for stereo videos).
        _left_distortion_coefficients (np.ndarray): The distortion coefficients for the left camera (for stereo videos).
        _right_calibration_matrix (np.ndarray): The calibration matrix for the right camera (for stereo videos).
        _right_distortion_coefficients (np.ndarray): The distortion coefficients for the right camera (for stereo videos).

    Methods:
        read(start=0, end=0): Read frames from the video.
        read_frame(frame_number): Read a single frame from the video.
        _read_mono(start=0, end=0): Read mono frames from the video.
        _read_stereo(start=0, end=0): Read stereo frames from the video.
        _read_mono_frame(frame_number): Read a single mono frame from the video.
        _read_stereo_frame(frame_number): Read a single stereo frame from the video.
        _return_numpy(frames): Convert frames to numpy array format.
        _return_tensor(frames): Convert frames to torch tensor format.
        _return_list(frames): Return frames as a list.
        _close(): Release the video capture object.
        _open_video(video_path): Open the video file using OpenCV.
        __len__(): Get the total number of frames in the video.
        video_width(): Get the width of the video frames.
        video_height(): Get the height of the video frames.
        left_fx(): Get the focal length in the x-direction for the left camera.
        left_fy(): Get the focal length in the y-direction for the left camera.
        left_cx(): Get the principal point in the x-direction for the left camera.
        left_cy(): Get the principal point in the y-direction for the left camera.
        right_fx(): Get the focal length in the x-direction for the right camera.
        right_fy(): Get the focal length in the y-direction for the right camera.
        right_cx(): Get the principal point in the x-direction for the right camera.
        right_cy(): Get the principal point in the y-direction for the right camera.
        baseline(): Get the baseline distance between the left and right cameras.
        left_calibration_mat(): Get the calibration matrix for the left camera.
        right_calibration_mat(): Get the calibration matrix for the right camera.
        stereo_calibration_mat(): Get the stereo calibration matrix.
        stereo_rotation_mat(): Get the stereo rotation matrix.
        stereo_translation_mat(): Get the stereo translation matrix.
    """
    def __init__(
        self,
        video_config: dict[str, VideoType | str | PosixPath],
        return_type: RetType | str = RetType.NUMPY,
    ) -> None:
        """
        Initialize the VideoReader object.

        Args:
            video_config (dict[str, VideoType | str | PosixPath]): A dictionary containing video configuration parameters.
            return_type (RetType | str, optional): The desired return type of the frames. Defaults to RetType.NUMPY.
        """
        
        video_path = video_config["video_path"]
        
        if not isinstance(video_path, PosixPath):
            video_path = Path(video_path)

        self.cap = self._open_video(video_path)
        self.video_path = video_path
        self.video_config = video_config
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_type = video_config["video_type"]
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._reading_method = self._read_mono if self.video_type == VideoType.MONO else self._read_stereo
        self._frame_reading_method = self._read_mono_frame if self.video_type == VideoType.MONO else self._read_stereo_frame
        self._return_func = {RetType.NUMPY: self._return_numpy, RetType.TENSOR: self._return_tensor, RetType.LIST: self._return_list}[return_type]
        
        try:
            self._left_calibration_matrix = np.array(video_config["left_calibration_matrix"][:9]).reshape(3, 3)
            self._left_distortion_coefficients = np.array(video_config["left_calibration_matrix"][9:])
            self._right_calibration_matrix = np.array(video_config["right_calibration_matrix"][:9]).reshape(3, 3)
            self._right_distortion_coefficients = np.array(video_config["right_calibration_matrix"][9:])
        
        except Exception:
            print("Calibration cofficients not found in metadata...")
            print("Starting reader without calibration params!")

    def _read_mono(
        self, 
        start: int = 0, 
        end: int = 0
    ) -> list[np.ndarray]:
        """
        Read and return a list of mono-channel frames from the video.

        Args:
            start (int): The starting frame number (default is 0).
            end (int): The ending frame number (default is 0, which means the end of the video).

        Returns:
            list[np.ndarray]: A list of mono-channel frames represented as numpy arrays.

        Raises:
            ValueError: If the end frame number is out of range for the video length.
        """
        if end == 0:
            end = self.video_length
        elif end > self.video_length:
            raise ValueError(f"End frame number out of range, for video of length {self.video_length}")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(start, end):
            ret, frame = self.cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                break
        return frames
    

    def _read_stereo(
        self, 
        start: int = 0, 
        end: int = 0
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Read stereo frames from the video.

        Args:
            start (int): The starting frame number (default: 0).
            end (int): The ending frame number (default: 0).

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing two lists of numpy arrays.
                The first list contains the left frames, and the second list contains the right frames.
        """
        if end == 0:
            end = self.video_length
        elif end > self.video_length:
            raise ValueError(f"End frame number out of range, for video of length {self.video_length}")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        left_frames, right_frames = [], []
        for _ in range(start, end):
            ret, frame = self.cap.read()
            if ret:
                half_width = self.width // 2
                left_frame = frame[:, :half_width, :]
                right_frame = frame[:, half_width:, :]
                left_frames.append(cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB))
                right_frames.append(cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB))
            else:
                break
        return left_frames, right_frames

    def read(
        self, 
        start: int = 0, 
        end: int = 0
    ) -> (np.ndarray | torch.Tensor) | tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        """
        Reads video frames from the specified start to end frame.

        Args:
            start (int): The starting frame index (default: 0).
            end (int): The ending frame index (default: 0).

        Returns:
            np.ndarray or torch.Tensor or tuple[np.ndarray or torch.Tensor, np.ndarray or torch.Tensor]:
                The video frames read from the specified range.
                If a mono video is read, it returns a single np.ndarray or torch.Tensor.
                If stereo video is read, it returns a tuple of np.ndarray or torch.Tensor,
                where the first element represents the left frames and the second element represents the right frames.
        """
        return self._return_func(self._reading_method(start, end))
    
    def read_frame(self, frame_number: int) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
            """
            Reads and returns a frame from the video.

            Args:
                frame_number (int): The number of the frame to be read.

            Returns:
                np.ndarray | tuple[np.ndarray, np.ndarray]: The frame read from the video.
                    If mono, returns a single numpy array.
                    If stereo, returns a tuple where the first element is the left frame and the second is the right frame.
            """
            return self._return_func(self._frame_reading_method(frame_number))
    
    def _read_mono_frame(
        self, 
        frame_number: int
    ) -> np.ndarray:
        """
        Read the frame from single camera the video at the specified frame number.

        Args:
            frame_number (int): The frame number to read.

        Returns:
            np.ndarray: The monochrome frame as a NumPy array.

        Raises:
            ValueError: If the frame number is out of range for the video length.
        """
        if frame_number > self.video_length:
            raise ValueError(f"Frame number out of range, for video of length {self.video_length}")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Frame number out of range, for video of length {self.video_length}")
    
    def _read_stereo_frame(
        self, 
        frame_number: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reads a stereo frame from the video file.

        Args:
            frame_number (int): The frame number to read.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the left and right stereo frames as numpy arrays.
        Raises:
            ValueError: If the frame number is out of range for the video length.
        """

        if frame_number > self.video_length:
            raise ValueError(f"End frame number out of range, for video of length {self.video_length}")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            half_width = self.width // 2
            left_frame = frame[:, :half_width, :]
            right_frame = frame[:, half_width:, :]
            return cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB), cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Frame number out of range, for video of length {self.video_length}")
    
    def _return_numpy(
        self, 
        frames: list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Convert a list or tuple of numpy arrays into a single numpy array or a tuple of numpy arrays.

        Args:
            frames (list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]): The input frames to be converted.

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray]: The converted numpy array or tuple of numpy arrays.
        """
        if isinstance(frames, tuple):
            return np.stack(frames[0]), np.stack(frames[1])
        else:
            return np.stack(frames)
        
    def _return_tensor(
        self, 
        frames: list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a list of numpy arrays or a tuple of lists of numpy arrays into torch tensors.

        Args:
            frames (list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]): The input frames to be converted.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: The converted torch tensors.

        Raises:
            None

        """
        if isinstance(frames, tuple):
            if isinstance(frames[0], np.ndarray):
                frames = (frames[0][np.newaxis, ...], frames[1][np.newaxis, ...])
                single_frame = True
            else:
                single_frame = False

            left_frames = [torch.tensor(frame).permute(2, 0, 1) for frame in frames[0]]
            right_frames = [torch.tensor(frame).permute(2, 0, 1) for frame in frames[1]]
            return torch.stack(left_frames).squeeze(0) if single_frame else torch.stack(left_frames), torch.stack(right_frames).squeeze(0) if single_frame else torch.stack(right_frames)
        else:
            if isinstance(frames, np.ndarray):
                frames = frames[np.newaxis, ...]
                single_frame = True
            else:
                single_frame = False
            frames = [torch.tensor(frame).permute(2, 0, 1) for frame in frames]
            return torch.stack(frames).squeeze(0) if single_frame else torch.stack(frames)
        
    def _return_list(
        self, 
        frames: list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]
    ) -> list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Returns the input frames as a list or tuple of lists.

        Args:
            frames (list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]): The input frames.

        Returns:
            list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]: The input frames as a list or tuple of lists.
        """
        return frames
    
    def _close(self) -> None:
        """
        Closes the video capture object.

        This method releases the video capture object, freeing up system resources.

        Returns:
            None
        """
        self.cap.release()
    
    def _open_video(self, video_path: PosixPath | str) -> cv2.VideoCapture:
        """
        Opens a video file using OpenCV and returns a VideoCapture object.

        Args:
            video_path (PosixPath | str): The path to the video file.

        Returns:
            cv2.VideoCapture: A VideoCapture object representing the opened video file.
        """
        return cv2.VideoCapture(str(video_path))
    
    def __len__(self) -> int:
        return self.video_length
    
    @property
    def video_width(self) -> int:
        return self.width
    
    @property
    def video_height(self) -> int:
        return self.height
    
    @property
    def left_fx(self) -> float:
        return self._left_calibration_matrix[0, 0]
    
    @property
    def left_fy(self) -> float:
        return self._left_calibration_matrix[1, 1]
    
    @property
    def left_cx(self) -> float:
        return self._left_calibration_matrix[0, 2]
    
    @property
    def left_cy(self) -> float:
        return self._left_calibration_matrix[1, 2]
    
    @property
    def right_fx(self) -> float:
        return self._right_calibration_matrix[0, 0]
    
    @property
    def right_fy(self) -> float:
        return self._right_calibration_matrix[1, 1]
    
    @property
    def right_cx(self) -> float:
        return self._right_calibration_matrix[0, 2]
    
    @property
    def right_cy(self) -> float:
        return self._right_calibration_matrix[1, 2]
    
    @property
    def baseline(self) -> float:
        return abs(self._stereo_calibration_matrix[3, 0])
    
    @property
    def left_calibration_mat(self) -> np.ndarray:
        return self._left_calibration_matrix
    
    @property
    def right_calibration_mat(self) -> np.ndarray:
        return self._right_calibration_matrix
    
    @property
    def stereo_calibration_mat(self) -> np.ndarray:
        return self._stereo_calibration_matrix
    
    @property
    def stereo_rotation_mat(self) -> np.ndarray:
        return self._stereo_calibration_matrix[:3, :3]
    
    @property
    def stereo_translation_mat(self) -> np.ndarray:
        return self._stereo_calibration_matrix[3, :3]
    