import pygame
from motion_spectrum import calculate_motion_spectrum, resize_spectrum_2_reference
from video_reader import VideoReader, RetType
from displacement import calculate_deformation_map
import torch
from complex import motion_spectrum_2_complex
import json
from pathlib import Path, PosixPath
from masking import Masking
import math
from torchvision.utils import flow_to_image
from optical_flow import warp_flow
import numpy as np
import torch.nn as nn
from enum import StrEnum
from dataclasses import dataclass, field
from pyOpenHaptics.hd_device import HapticDevice
import pyOpenHaptics.hd as hd
from pyOpenHaptics.hd_callback import hd_callback


@dataclass
class DeviceState:
    button: bool = False
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
    device_state.position = [transform[3][0], -transform[3][1], transform[3][2]]
    # Set the force to the device
    hd.set_force(device_state.force)
    # Get the current state of the device buttons
    button = hd.get_buttons()
    device_state.button = True if button == 1 else False

from depth import calculate_depth_map, load_depth_model_and_transform, apply_depth_culling, ModelType

rad = math.pi / 180

class ControlType(StrEnum):
    MOUSE = "mouse"
    HAPTIC = "haptic"


class InteractiveDemo(object):
    def __init__(
        self,
        video_path: PosixPath | str,
        start: int = 0,
        end: int = 0,
        K: int = 16,
        depth_model_type: ModelType | str = ModelType.DPT_Large,
        control_type: str = ControlType.MOUSE,
        filtering: bool = True,
        masking: bool = True
    ) -> None:

        frames = self._init_video_reader(video_path, start, end)
        self.reference_frame = self.video_reader.read_frame(0)[0]
        self.depth_map = self._get_depth_map(self.reference_frame, depth_model_type)
        
        # Normalize depth map for depth culling
        self.depth_map = (self.depth_map - self.depth_map.min()) / (self.depth_map.max() - self.depth_map.min())
        display_frame = apply_depth_culling(self.depth_map[0], self.reference_frame[0])
        self._get_motion_spectrum_from_video(frames, K, filtering, masking)
        self._control_func = {ControlType.MOUSE: self.mouse_control, ControlType.HAPTIC: self.haptic_control}[control_type]
        self.control_type = control_type
        self._init_pygame(display_frame)
        
        if control_type == ControlType.HAPTIC:
            self._init_haptic_device()
            

    def _init_pygame(
        self,
        reference_frame: np.ndarray
    ) -> None:
        """
        Initialize the pygame window and display the reference frame.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((reference_frame.shape[1] - 30, reference_frame.shape[0] - 30))
        display_frame = self._crop_image_for_display(reference_frame, (reference_frame.shape[0] - 30, reference_frame.shape[1] - 30))
        self.image = pygame.transform.rotate(pygame.transform.flip(pygame.surfarray.make_surface(display_frame), False, True), -90)
        pygame.display.set_caption("Interactive Demo")
        
        self.screen.blit(self.image, (0, 0))
        pygame.display.flip()
        self.running = True
        self.on_click = False
        self.pixel = None
        self.cur_pos = None
        self.clock = pygame.time.Clock()
        self.displacement = torch.tensor([0, 0])
    
    def _init_haptic_device(
        self
    ) -> None:
        """
        Initialize the haptic device.
        """
        device_state = DeviceState()
        self.device = HapticDevice(device_name="Default Device", callback=device_callback)
    
    def haptic_control(
        self
    ) -> None:
        def cursor(screen, color, pos, radius):
            pygame.draw.circle(screen, color, pos, radius)
        
        pos_x, pos_y = device_state.position[0], device_state.position[1]

        cursor(self.screen, (255, 255, 255), (int(pos_x), int(pos_y)), 10)
            
        print(device_state.position)
        if device_state.button:
            print("Button pressed")
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
            if event.type == pygame.MOUSEBUTTONUP:
                self.on_click = False
            if event.type == pygame.MOUSEMOTION and self.on_click:
                self.displacement = torch.tensor([event.pos[0] - self.pixel[0], event.pos[1] - self.pixel[1]])
                self.alpha = min(15.0, self.displacement.float().norm(2) / 10)
                self.displacement = self.displacement.float() / self.displacement.float().norm(2)
                self.cur_pos = event.pos
            if event.type == pygame.QUIT:
                self.running = False

    def sim_step(
        self,
    ) -> None:
        """
        Simulate the deformation of the frame based on the user input.
        """
        self.clock.tick(30)
        self.screen.fill((1, 1, 1))
        
        # Control the deformation of the frame
        self._control_func()
        # Update the rendering of the frame
        self._update_render()

        pygame.display.flip()
    
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
        
        if self.cur_pos is not None and self.pixel is not None and self.on_click:
            deformation_map, _ = calculate_deformation_map(self.motion_spectrum[0], -self.displacement, self.pixel, self.alpha)
            deformed_frame = warp_flow(torch.from_numpy(self.reference_frame).permute(2, 0, 1).float().cuda(), deformation_map.cuda(), self.depth_map, self.mask)
            deformed_frame = self._crop_image_for_display(deformed_frame, (self.reference_frame.shape[0] - 30, self.reference_frame.shape[1] - 30))
            deformed_frame_surf = pygame.transform.rotate(pygame.transform.flip(pygame.surfarray.make_surface(deformed_frame), False, True), -90)
            self.screen.blit(deformed_frame_surf, (0, 0))
            arrow(self.screen, (255, 255, 255), (255, 255, 255), self.pixel, self.cur_pos, 10)
        else:
            self.screen.blit(self.image, (0, 0))
    
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
        depth_model_type: ModelType | str
    ) -> np.ndarray:
        """
        Initialize the depth model and calculate the depth map for the reference frame.
        """
        model, transform = load_depth_model_and_transform(depth_model_type)
        model.eval()
        model.to("cuda")
        depth_map = calculate_depth_map(model, transform, reference_frame)
        return depth_map

    def _get_motion_spectrum_from_video(
        self,
        frames: np.ndarray | tuple[np.ndarray, np.ndarray],
        K: int,
        filtering: bool,
        masking: bool
    ) -> None:
        """
        Get the motion spectrum from the video frames.
        """
        if masking:
            mask = self._init_masking()
            self.mask = mask.masking
        else:
            mask = None
            self.mask = None
        if isinstance(frames, tuple):
            motion_spectrum = (calculate_motion_spectrum(frames[0], K, filtered=filtering, mask=mask, camera_pos="left", save_flow_video=True), calculate_motion_spectrum(frames[1], K, filtered=filtering, mask=mask, camera_pos="right"))
            motion_spectrum = resize_spectrum_2_reference(motion_spectrum, self.reference_frame)
            self.motion_spectrum, _ = motion_spectrum_2_complex(motion_spectrum)
        else:
            motion_spectrum = calculate_motion_spectrum(frames, K, filtered=filtering, mask=mask)
            self.motion_spectrum = motion_spectrum_2_complex(resize_spectrum_2_reference(motion_spectrum, self.reference_frame)) 
        
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


if __name__ == "__main__":
    pygame_demo = InteractiveDemo("videos/liver_stereo.avi", control_type=ControlType.MOUSE, filtering=True, masking=True)
    pygame_demo.run()

