from pymodal_surgical.apps.utils import ModeShapeCalculator
from pymodal_surgical.video_processing.writer import VideoWriter
from pymodal_surgical.modal_analysis import solver, deformation, depth, optical_flow
import numpy as np
import torch
import random

default_values = {
    "time": 10.0,
    "pixel": [10, 10],
    "force": [10, 0],
    "mass": 1.0,
    "rayleigh_mass": 0.01,
    "rayleigh_stiffness": 0.01,
    "dt": 0.01
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VideoGenerator:
    def __init__(self, config: dict):
        
        mode_shape_config = config["mode_shape_calculator"]
        self._intialize_synthesizer(mode_shape_config)
        self.video_writer = VideoWriter(config["video_writer"])
        video_generator_config = config["video_generator"]
        self._parse_interaction_command(video_generator_config)

        
    def _sim_step(
        self,
        pixel: int,
        force_vector: np.ndarray,
        modal_coordinate: torch.Tensor,
        solving: bool = False
    ) -> np.ndarray:
        
        if solving:
            modal_coordinate = solver.euler_solver(
                modal_coordinate, 
                self.K, 
                force_vector, 
                self.mass, 
                pixel, 
                alpha=self.rayleigh_mass, 
                beta=self.rayleigh_stiffness, 
                time_step=self.dt
            )

            displacement = deformation.calculate_deformation_map_from_modal_coordinate(
                self.mode_shapes,
                modal_coordinate
            )
        
        else:
            displacement, modal_coordinate = deformation.calculate_deformation_map_from_displacement(
                self.mode_shapes, 
                force_vector, 
                pixel
            )
        
        return displacement, modal_coordinate
    
    def generate_video(self):
        frames = []
        modal_coordinate = torch.zeros(self.K, 2)

        for idx, t in enumerate(self.time):
            try:
                current_force = self.force[idx]
                solving = False
            except IndexError:
                current_force = 0.0
                solving = True
            displacement, modal_coordinate = self._sim_step(
                self.pixel, current_force, modal_coordinate, solving=solving
            )

            frame = optical_flow.warp_flow(self.reference_frame, displacement, self.depth_map, self.mask)
            
            frames.append(frame)

        self.video_writer(frames)


    def _intialize_synthesizer(self, mode_shape_config: dict):
        
        mode_shape_calculator = ModeShapeCalculator(mode_shape_config)
        self.mode_shapes = mode_shape_calculator.complex_mode_shapes
        self.K = mode_shape_calculator.K
        self.reference_frame = mode_shape_calculator.frames[0]
        self.mask = mode_shape_calculator.mask
        depth_model, depth_transform = depth.load_depth_model_and_transform(device=device)
        self.depth_map = depth.calculate_depth_map(depth_model, depth_transform, self.reference_frame, device=device)


    def _parse_interaction_command(self, video_generator_config: dict) -> None:

        expected_keys = default_values.keys()

        for key in expected_keys:
            if key not in video_generator_config:
                video_generator_config[key] = default_values[key]
        
        self.dt = video_generator_config["dt"]
        self.time = np.arange(0, video_generator_config["time"], self.dt)
        self.pixel = video_generator_config["pixel"]
        self.force = torch.arange(0, video_generator_config["time"] / 2, self.dt).unsqueeze(1) * torch.tensor(video_generator_config["force"])
        self.mass = video_generator_config["mass"]
        self.rayleigh_mass = video_generator_config["rayleigh_mass"]
        self.rayleigh_stiffness = video_generator_config["rayleigh_stiffness"]
        