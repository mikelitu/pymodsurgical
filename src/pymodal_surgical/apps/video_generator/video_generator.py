from pymodal_surgical.apps.utils import ModeShapeCalculator
from pymodal_surgical.video_processing.writer import VideoWriter
from pymodal_surgical.modal_analysis import solver, deformation, depth, optical_flow
import numpy as np
import torch
from pathlib import Path

default_values = {
    "video_path": ".",
    "video_type": "mono",
    "time": 10.0,
    "interact": 0.5,
    "pixel": [10, 10],
    "force": [10, 0],
    "crop": [0, 0],
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
        
        video_generator_config = config["video_generator"]
        self._parse_interaction_command(video_generator_config)

        video_writer_config = self._generate_video_writer_config(video_generator_config)
        self.video_writer = VideoWriter(video_writer_config)
        
        
    def _sim_step(
        self,
        pixel: int,
        force_vector: np.ndarray,
        modal_coordinate: torch.Tensor,
        solving: bool = False
    ) -> np.ndarray:
        
        if solving:
            
            force_vector = (force_vector / self.K) * torch.ones([self.K, 2], dtype=torch.cfloat)

            if len(modal_coordinate.shape) == 4:
                modal_coordinate = modal_coordinate.squeeze(-1).squeeze(-1)
                
            modal_coordinate = solver.euler_solver(
                modal_coordinate, 
                self.frequencies,
                self.mass, 
                force_vector, 
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
        modal_coordinate = torch.zeros(self.K, 2, dtype=torch.cfloat)

        for idx, t in enumerate(self.time):
            try:
                current_force = self.force[idx]
                solving = True
            
            except IndexError:
                current_force = 0.0
                solving = True
            
            displacement, modal_coordinate = self._sim_step(
                self.pixel, current_force, modal_coordinate, solving=solving
            )

            # print(displacement.shape, self.reference_frame.shape, self.mask.shape)
            frame = optical_flow.warp_flow(self.reference_frame, displacement.to(device), self.depth_map, self.mask)
            frame = self._crop_display(frame, self.crop)
            frames.append(frame)

        self.video_writer(frames)


    def _crop_display(self, frame: np.ndarray, crop: list) -> np.ndarray:
        if crop[0] == 0 and crop[1] == 0:
            return frame
        elif crop[0] == 0:
            return frame[:, crop[1]:-crop[1], :]
        elif crop[1] == 0:
            return frame[crop[0]:-crop[0], :, :]
        
        return frame[crop[0]:-crop[0], crop[1]:-crop[1], :]
    
    
    def _intialize_synthesizer(self, mode_shape_config: dict):
        
        mode_shape_calculator = ModeShapeCalculator(mode_shape_config)
        self.mode_shapes = mode_shape_calculator.complex_mode_shapes
        self.K = mode_shape_calculator.K
        self.frequencies = mode_shape_calculator.frequencies
        reference_frame = mode_shape_calculator.frames[0]
        self.mask = mode_shape_calculator.mask.masking
        depth_model, depth_transform = depth.load_depth_model_and_transform(device=device)
        self.depth_map = depth.calculate_depth_map(depth_model, depth_transform, reference_frame, device=device)
        self.depth_map =(self.depth_map - self.depth_map.min()) / (self.depth_map.max() - self.depth_map.min())
        self.reference_frame = torch.from_numpy(reference_frame).to(device).permute(2, 0, 1).unsqueeze(0)

    def _parse_interaction_command(self, video_generator_config: dict) -> None:

        expected_keys = default_values.keys()

        for key in expected_keys:
            if key not in video_generator_config:
                video_generator_config[key] = default_values[key]
        
        self.dt = video_generator_config["dt"]
        self.time = np.arange(0, video_generator_config["time"], self.dt)
        self.pixel = video_generator_config["pixel"]
        self.force = torch.linspace(0, 1, int((video_generator_config["time"] * video_generator_config["interact"]) / video_generator_config["dt"])).unsqueeze(1) * torch.tensor(video_generator_config["force"])
        self.mass = video_generator_config["mass"] * torch.ones(self.K, dtype=torch.cfloat)
        self.rayleigh_mass = video_generator_config["rayleigh_mass"]
        self.rayleigh_stiffness = video_generator_config["rayleigh_stiffness"]
        self.crop = video_generator_config["crop"]
    
    def _generate_video_writer_config(self, video_generator_config: dict) -> dict:
        
        video_type = video_generator_config["video_type"]
        u_force = video_generator_config["force"][0]
        v_force = video_generator_config["force"][1]

        video_path = Path(video_generator_config["video_path"])/f"{u_force}_{v_force}_{video_type}.mp4"
        fps = 1 / self.dt
        
        return {
            "video_path": str(video_path),
            "video_type": video_type,
            "fps": fps
        }