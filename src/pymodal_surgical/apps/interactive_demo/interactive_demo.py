import pygame

from pymodal_surgical.modal_analysis import deformation, math_helper, optical_flow, motion, depth, solver
from pymodal_surgical.modal_analysis import functions
from pymodal_surgical.video_processing.reader import VideoReader, RetType
import torch
from pathlib import Path, PosixPath
from pymodal_surgical.video_processing.masking import Masking
import math
import numpy as np
from enum import Enum

try:
    from dataclasses import dataclass, field
    from pyOpenHaptics.hd_device import HapticDevice
    import pyOpenHaptics.hd as hd
    from pyOpenHaptics.hd_callback import hd_callback
except OSError:
    pass

import time
import argparse
import json


rad = math.pi / 180
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

try:
    @dataclass
    class DeviceState:
        button: bool = False
        pre_button: bool = False
        position: list = field(default_factory=list)
        force: list = field(default_factory=list)

    device_state = DeviceState()

    @hd_callback
    def device_callback():
        global device_state
        """
        Callback function for the haptic device.
        """
        # Get the current position of the device
        transform = hd.get_transform()
        device_state.position = [transform[3][0], transform[3][1], transform[3][2]]
        # Set the force to the device
        hd.set_force(device_state.force)
        # Get the current state of the device buttons
        button = hd.get_buttons()
        device_state.button = True if button == 1 else False

except NameError:
    pass

class ControlType(Enum, str):
    MOUSE = "mouse"
    HAPTIC = "haptic"


class InteractiveDemo(object):
    
    def __init__(
        self,
        video_path: PosixPath | str,
        solver_time_step: float = 0.03,
        fps: int = 24,
        start: int = 0,
        end: int = 0,
        K: int = 16,
        depth_model_type: depth.ModelType | str = depth.ModelType.DPT_Large,
        control_type: str = ControlType.MOUSE,
        norm_scale: float = 10.0,
        alpha_limit: float = 5.0,
        force_scale: float = 0.35,
        display_cropping: tuple[int, int] = (30, 30),
        filtering: bool = True,
        masking: bool = True,
        near: float = 0.05,
        far: float = 0.95,
        mass: float = 1.0,
        rest_force: float = 10.0,
        inverse: bool = False,
        rayleigh_mass: float = 0.5,
        rayleigh_stiffness: float = 0.5
    ) -> None:

        frames = self._init_video_reader(video_path, start, end)
        
        if isinstance(frames, tuple):
            self.reference_frame = self.video_reader.read_frame(0)[0]
        else:
            self.reference_frame = self.video_reader.read_frame(0)

        self.depth_map = self._get_depth_map(self.reference_frame, depth_model_type)
        self.near, self.far = near, far
        self.inverse = inverse
        # Normalize depth map for depth culling
        self.depth_map = (self.depth_map - self.depth_map.min()) / (self.depth_map.max() - self.depth_map.min())
        display_frame = depth.apply_depth_culling(self.depth_map[0], self.reference_frame, self.near, self.far, self.inverse)
        self._get_mode_shapes_from_video(frames, K, filtering, masking)
        self._control_func = {ControlType.MOUSE: self.mouse_control, ControlType.HAPTIC: self.haptic_control}[control_type]
        self.control_type = control_type
        if control_type == ControlType.HAPTIC:
            self._init_haptic_device()
            self.haptic_limits = torch.tensor([[-210.0, 216.0], [27.7, 330.0]])
            self.force_limits = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]) 
            self.force_scale = force_scale
        
        
        self.display_cropping = display_cropping
        self._init_pygame(display_frame)
        self.alpha_limit = alpha_limit
        self.norm_scale = norm_scale
        timesteps = len(frames[0]) if isinstance(frames, tuple) else len(frames)
        self.frequencies = motion.get_motion_frequencies(timesteps, K, 1 / fps)
        self.solve = False
        self.fps = fps
        self.solver_time_step = solver_time_step
        self.rayleigh_mass = rayleigh_mass
        self.rayleigh_stiffness = rayleigh_stiffness
        self.mass = mass * torch.ones(self.mode_shapes.shape[0]).to(device)
        self.rest_force = rest_force * torch.ones(self.mode_shapes.shape[0], 2).to(device, dtype=torch.cfloat)

    def _init_pygame(
        self,
        reference_frame: np.ndarray
    ) -> None:
        """
        Initialize the pygame window and display the reference frame.
        """
        pygame.init()
        pygame.mouse.set_visible(False)
        self.screen = pygame.display.set_mode((reference_frame.shape[1] - self.display_cropping[1], reference_frame.shape[0] - self.display_cropping[0]))
        display_frame = self._crop_image_for_display(reference_frame, (reference_frame.shape[0] - self.display_cropping[0], reference_frame.shape[1] - self.display_cropping[1]))
        self.image = pygame.transform.rotate(pygame.transform.flip(pygame.surfarray.make_surface(display_frame), False, True), -90)
        pygame.display.set_caption("Interactive Demo")
        
        self.screen.blit(self.image, (0, 0))
        pygame.display.flip()
        self.running = True
        self.on_click = False
        self.pixel = None
        self.cur_pos = [0, 0]
        self.alpha = 0
        self.clock = pygame.time.Clock()
        self.displacement = torch.tensor([0, 0])
        

    def _init_haptic_device(
        self
    ) -> None:
        """
        Initialize the haptic device.
        """
        global device_state
        device_state = DeviceState()
        self.device = HapticDevice(device_name="Default Device", callback=device_callback)
        # Let the callback run for a bit to get the initial state of the device
        time.sleep(0.1)
        # Start a loop so the user sets the desired intial position
        print("Set the initial position of the device. Move the device to the following position: [0., 175.]")
        
        precision_range = 1.0
        self.backoff_count = 0

        try:
            while True:
                print("Current position: ", device_state.position[:2])
                diff_x = abs(device_state.position[0] + 0)
                diff_y = abs(device_state.position[1] - 175.0)
                if diff_x < precision_range and diff_y < precision_range:
                    self.init_position = torch.tensor(device_state.position[:2])
                    self.pre_button = False
                    break
                
                time.sleep(0.1)
        
        except Exception as e:
            self.backoff_count += 1
            print("Connecting to the device...")
            time.sleep(0.5)
            if self.backoff_count == 10:
                print("An exception occurred. Closing the device...")
                self.device.close()
                raise(e)


    def _scale_haptic_2_screen(
        self,
        position: torch.Tensor
    ) -> tuple[int, int]:
        """
        Scale the haptic position to the screen size.
        """
        width, height = pygame.display.get_surface().get_size()
        pos_x = ((position[0] - self.haptic_limits[0][0]) / (self.haptic_limits[0][1] - self.haptic_limits[0][0])) * width
        pos_y = height - ((position[1] - self.haptic_limits[1][0]) / (self.haptic_limits[1][1] - self.haptic_limits[1][0])) * height
        return int(np.clip(pos_x, 0, width)), int(np.clip(pos_y, 0, height))


    def haptic_control(
        self
    ) -> None:
        """
        Control the deformation of the frame using a haptic device
        """

        position = torch.tensor(device_state.position[:2])
        self.cur_pos = self._scale_haptic_2_screen(position)
        
        if device_state.button and not self.pre_button:
            self.on_click = True
            self.init_position = position
            self.displacement = torch.tensor([0, 0])
            self.alpha = 0
            self.pixel = self._scale_haptic_2_screen(position)
        
        elif device_state.button and self.pre_button:
            movement = position - self.init_position
            self.displacement = movement / (movement.float().norm(2) + 1e-8)
            self.displacement[1] = -self.displacement[1]
            self.alpha = min(self.alpha_limit, movement.float().norm(2) / self.norm_scale)
            # Set a force depending on the increase in the displacement
            cur_force_scale = self.alpha * self.force_scale
            force = cur_force_scale * np.array([-self.displacement[0], self.displacement[1], 0.0])
            # self.cur_pos = self._scale_haptic_2_screen(position)
            device_state.force = list(force)
        
        elif not device_state.button and self.pre_button:
            self.on_click = False
            self.pixel = None
            self.displacement = torch.tensor([0, 0])
            self.alpha = 0
            device_state.force = [0.0, 0.0, 0.0]

        self.pre_button = device_state.button
        
        for event in pygame.event.get():
            self._check_exit(event)


    def _check_exit(
        self,
        event: pygame.event.Event
    ) -> None:
        """
        Check if the user wants to exit the program.
        """
        if event.type == pygame.QUIT:
            self.running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False


    def mouse_control(
        self
    ) -> None:
        """
        Control the deformation of the frame using the mouse.
        """
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.on_click = True
                self.pixel = event.pos
                self.cur_pos = event.pos
                self.displacement = torch.tensor([0, 0])
                self.alpha = 0
                self.solve = False
            
            if event.type == pygame.MOUSEBUTTONUP:
                self.on_click = False
                self.solve = True
                self.calc_pixel = self.pixel

                if isinstance(self.mode_shapes, tuple):
                    self.modal_coordinates = functions.calculate_modal_coordinate(self.mode_shapes[0], -self.displacement, self.calc_pixel, self.alpha)
                else:
                    self.modal_coordinates = functions.calculate_modal_coordinate(self.mode_shapes, -self.displacement, self.calc_pixel, self.alpha)
            
            if event.type == pygame.MOUSEMOTION:
                if self.on_click:
                    self.displacement = torch.tensor([event.pos[0] - self.pixel[0], event.pos[1] - self.pixel[1]])
                    self.alpha = min(self.alpha_limit, self.displacement.float().norm(2) / self.norm_scale)
                    self.displacement = self.displacement.float() / self.displacement.float().norm(2)

                self.cur_pos = event.pos
            
            self._check_exit(event)


    def sim_step(
        self,
    ) -> None:
        """
        Simulate the deformation of the frame based on the user input.
        """
        self.clock.tick(self.fps)
        self.screen.fill((1, 1, 1))
        
        # Control the deformation of the frame
        self._control_func()
        # Update the rendering of the frame
        self._update_render()
        # Update the display
        pygame.display.flip()


    def _solve_ode(
        self
    ) -> torch.Tensor:
        """
        Solve the ODE for the modal coordinates.
        """

        self.modal_coordinates = solver.euler_solver(
            self.modal_coordinates, 
            self.frequencies, 
            self.mass, 
            self.rest_force, 
            alpha=self.rayleigh_mass, 
            beta=self.rayleigh_stiffness, 
            time_step=self.solver_time_step
        )
        
        if isinstance(self.mode_shapes, tuple):
            deformation_map = deformation.calculate_deformation_map_from_modal_coordinate(self.mode_shapes[0], self.modal_coordinates)
        else:
            deformation_map = deformation.calculate_deformation_map_from_modal_coordinate(self.mode_shapes, self.modal_coordinates)

        return deformation_map


    def _update_render(
        self
    ) -> None:
        """
        Update the rendering of the frame.
        """
        def arrow(screen, lcolor, tricolor, start, end, trirad, thickness=10):
            pygame.draw.line(screen, lcolor, start, end, thickness)
            rotation = (math.atan2(start[1] - end[1], end[0] - start[0])) + math.pi/2
            pygame.draw.polygon(screen, tricolor, ((end[0] + trirad * math.sin(rotation),
                                                end[1] + trirad * math.cos(rotation)),
                                            (end[0] + trirad * math.sin(rotation - 120*rad),
                                                end[1] + trirad * math.cos(rotation - 120*rad)),
                                            (end[0] + trirad * math.sin(rotation + 120*rad),
                                                end[1] + trirad * math.cos(rotation + 120*rad))))
        
        def cursor(screen, color, pos, radius = 10):
            pygame.draw.circle(screen, color, pos, radius)

        if self.pixel is not None and self.on_click:
            if isinstance(self.mode_shapes, tuple):
                deformation_map, _ = deformation.calculate_deformation_map_from_displacement(self.mode_shapes[0], -self.displacement, self.pixel, self.alpha)
            else:
                deformation_map, _ = deformation.calculate_deformation_map_from_displacement(self.mode_shapes, -self.displacement, self.pixel, self.alpha)
            
            deformed_frame = optical_flow.warp_flow(torch.from_numpy(self.reference_frame).permute(2, 0, 1).float().to(device), deformation_map.to(device), self.depth_map, self.mask, self.near, self.far, self.inverse)
            deformed_frame = self._crop_image_for_display(deformed_frame, (self.reference_frame.shape[0] - self.display_cropping[0], self.reference_frame.shape[1] - self.display_cropping[1]))
            deformed_frame_surf = pygame.transform.rotate(pygame.transform.flip(pygame.surfarray.make_surface(deformed_frame), False, True), -90)
            self.screen.blit(deformed_frame_surf, (0, 0))
            
            arrow(self.screen, (255, 255, 255), (255, 255, 255), self.pixel, self.cur_pos, 10)
        
        elif self.solve:
            deformation_map = self._solve_ode()
            deformed_frame = optical_flow.warp_flow(torch.from_numpy(self.reference_frame).permute(2, 0, 1).float().to(device), deformation_map.to(device), self.depth_map, self.mask, self.near, self.far, self.inverse)
            deformed_frame = self._crop_image_for_display(deformed_frame, (self.reference_frame.shape[0] - self.display_cropping[0], self.reference_frame.shape[1] - self.display_cropping[1]))
            deformed_frame_surf = pygame.transform.rotate(pygame.transform.flip(pygame.surfarray.make_surface(deformed_frame), False, True), -90)
            self.screen.blit(deformed_frame_surf, (0, 0))

        else:
            self.screen.blit(self.image, (0, 0))
        
        cursor(self.screen, (255, 255, 255), self.cur_pos, 10)
    

    def _init_masking(
        self
    ) -> Masking:
        """
        Initialize the mask for the video.
        """
        return Masking(self.video_reader.video_config[self.video_reader.video_path.stem]["mask"], self.video_reader.video_type)
    

    def _init_video_reader(
        self,
        video_path: PosixPath | str,
        start: int,
        end: int
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Initialize the video reader and read the frames.
        """
        if isinstance(video_path, str):
            video_path = Path(video_path)
        
        with open("videos/metadata.json", "r") as f:
            video_config = json.load(f)
        
        self.video_reader = VideoReader(video_path, video_config, return_type=RetType.NUMPY)
        return self.video_reader.read(start, end)
    

    def _get_depth_map(
        self,
        reference_frame: np.ndarray,
        depth_model_type: depth.ModelType | str
    ) -> np.ndarray:
        """
        Initialize the depth model and calculate the depth map for the reference frame.
        """
        model, transform = depth.load_depth_model_and_transform(depth_model_type)
        model.eval()
        model.to(device)
        depth_map = depth.calculate_depth_map(model, transform, reference_frame, device)
        return depth_map


    def _get_mode_shapes_from_video(
        self,
        frames: np.ndarray | tuple[np.ndarray, np.ndarray],
        K: int,
        filtering: bool,
        masking: bool
    ) -> None:
        """
        Get the mode shapesfrom the video frames.
        """
        if masking:
            mask = self._init_masking()
            self.mask = mask.masking
        else:
            mask = None
            self.mask = None
        if isinstance(frames, tuple):
            mode_shapes = (functions.calculate_mode_shapes(frames[0], K, filtered=filtering, mask=mask, camera_pos="left", save_flow_video=False), functions.calculate_mode_shapes(frames[1], K, filtered=filtering, mask=mask, camera_pos="right"))
            mode_shapes = functions.resize_spectrum_2_reference(mode_shapes, self.reference_frame)
            self.mode_shapes, _ = math_helper.mode_shape_2_complex(mode_shapes)
        else:
            mode_shapes = functions.calculate_mode_shapes(frames, K, filtered=filtering, mask=mask)
            self.mode_shapes = math_helper.mode_shape_2_complex(functions.resize_spectrum_2_reference(mode_shapes, self.reference_frame)) 
    

    def _crop_image_for_display(
        self,
        image: np.ndarray,
        crop_size: tuple[int, int]
    ) -> np.ndarray:
        """
        Crop the image to the specified size, so in the rendering looks better.
        """
        width, height = image.shape[1], image.shape[0]
        if width < crop_size[1] or height < crop_size[0]:
            raise ValueError("The crop size is larger than the image")
        diff_width = width - crop_size[1]
        diff_height = height - crop_size[0]
        return image[diff_height // 2:height - diff_height // 2, diff_width // 2:width - diff_width // 2]


    def run(
        self
    ) -> None:
        """
        Run the interactive demo.
        """
        while self.running:
            self.sim_step()
        
        pygame.quit()
        
        if self.control_type == ControlType.HAPTIC:
            self.device.close()

