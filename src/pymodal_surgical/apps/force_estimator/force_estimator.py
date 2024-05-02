import torch
import numpy as np
from pathlib import Path
from pymodal_surgical.apps.utils import ModeShapeCalculator
from pymodal_surgical.video_processing.reader import VideoReader
from pymodal_surgical.modal_analysis.depth import load_depth_model_and_transform, calculate_depth_map, ModelType
import pymodal_surgical.modal_analysis.optical_flow as optical_flow
import pymodal_surgical.modal_analysis.force as force
import pymodal_surgical.modal_analysis.math_helper as math_helper
import matplotlib.pyplot as plt


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class ForceEstimator():

    def __init__(
        self,
        force_estimation_config: dict,
        mode_shape_config: dict,
    ) -> None:
        
        mode_shape_calculator = ModeShapeCalculator(mode_shape_config)

        self.mode_shapes = mode_shape_calculator.complex_mode_shapes
        self.frequencies = mode_shape_calculator.frequencies
        self._load_force_video(force_estimation_config)
    
    def _load_force_video(
        self,
        force_estimation_config: dict
    ) -> VideoReader:
        
        self.force_video_reader = VideoReader(video_config=force_estimation_config)
        self.flow_model = optical_flow.load_flow_model(device)

    
    def _show_frames(
        self,
        frame: np.ndarray,
    ) -> None:
        plt.imshow(frame)
        plt.show()
        
    
    def _calculate_force(
        self,
        idx_1: int,
        idx_2: int,
        pixels: tuple[int, int],
    ) -> torch.Tensor:
        
        # Load the frames
        frame_1 = self.force_video_reader.read_frame(idx_1)
        frame_2 = self.force_video_reader.read_frame(idx_2)

        # Preprocess the frames for the calculation of optical flow
        frame_1 = optical_flow.preprocess_for_raft(frame_1).unsqueeze(0).to(device)
        frame_2 = optical_flow.preprocess_for_raft(frame_2).unsqueeze(0).to(device)

        # Calculate the optical flow
        flow = optical_flow.estimate_flow(self.flow_model, frame_1, frame_2)
        flow = flow.squeeze(0).detach().cpu()

        # The optical flow corresponds to the desired displacement of the pixels
        # We need to calculate the force that is applied to the pixels
        # We can do this by calculating the constraint force
        # The constraint force is the force that is applied to the pixels to keep them in place

        # Calculate the constraint force
        modal_coordinates = torch.zeros(self.mode_shapes.shape[0], 2)
        norm_mode_shapes = math_helper.orthonormal_normalization(self.mode_shapes)
        constraint_force = force.calculate_force_from_displacement_map(norm_mode_shapes, flow, modal_coordinates, self.frequencies, pixels, timestep=0.03)
        print(constraint_force)
        pass


if __name__ == "__main__":
    mode_shape_config = {
        "video_path": "C:\\Users\\md21\\surgical-video-modal-analysis\\videos\\20240430_041703.mp4",
        "K": 16,
        "fps": 20.0,
        "video_type": "mono",
        "start": 0,
        "end": 0,
        "masking": {
            "enabled": True,
            "mask": "C:\\Users\\md21\\surgical-video-modal-analysis\\videos\\mask\\heart_beating.png"
        },
        "filtering": {
            "enabled": True,
            "size": 13,
            "sigma": 3.0
        }
    }
    force_video_config = {
        "fps": 20.0,
        "video_type": "mono",
        "video_path": "C:/Users/md21/surgical-video-modal-analysis/videos/20240430_041703.mp4",
    }

    estimator = ForceEstimator(
        force_estimation_config=force_video_config,
        mode_shape_config=mode_shape_config
    )

    estimator._calculate_force(270, 360, (10, 30))