from . import depth, optical_flow, motion, math_helper
import torch
import numpy as np
from ..video_processing.filtering import GaussianFiltering
from ..video_processing.masking import Masking
from ..video_processing.writer import VideoWriter

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = optical_flow.load_flow_model(device)


def calculate_mode_shapes(
    frames: list[np.ndarray], 
    K: int,
    depth_maps: list[np.ndarray] | None = None,
    batch_size: int = 50, 
    filter_config: dict | None = None,
    mask: Masking | None = None,
    camera_pos: str | None = None,
    save_flow_video: bool = False,
) -> torch.Tensor:
    """
    Calculates the mode shapes from the given frames using optical flow.

    Args:
        frames (list[np.ndarray]): List of frames as numpy arrays.
        K (int): Number of mode shapes to calculate.
        depth_maps (list[np.ndarray] | None, optional): List of depth maps as numpy arrays. Defaults to None.
        batch_size (int, optional): Batch size for estimating optical flow. Defaults to 500.
        filter_config (dict | None, optional): Configuration for Gaussian filtering. Defaults to None.
        mask (Masking | None, optional): Masking object for applying masks to mode shapes. Defaults to None.
        camera_pos (str | None, optional): Camera position information. Defaults to None.
        save_flow_video (bool, optional): Flag to save the optical flow video. Defaults to False.

    Returns:
        torch.Tensor: Tensor containing the calculated mode shapes.
    """
    
    preprocess_frames = [optical_flow.preprocess_for_raft(frame) for frame in frames]
    preprocess_frames = torch.stack(preprocess_frames).to(device)

    B = preprocess_frames.shape[0]
    flows = torch.zeros((B - 1, 2, preprocess_frames.shape[2], preprocess_frames.shape[3])).to(device)

    reference_frames = preprocess_frames[:-1]
    target_frames = preprocess_frames[1:]

    if B > batch_size:
        number_of_batches = B // batch_size
        for i in range(number_of_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            flows[start:end] = optical_flow.estimate_flow(model, reference_frames[start:end], target_frames[start:end]).squeeze(0)
    else:
        flows = optical_flow.estimate_flow(model, reference_frames, target_frames).squeeze(0)

    if filter_config is not None:
        if filter_config["enabled"]:
            filtering = GaussianFiltering(filter_config["size"], filter_config["sigma"])
            flows = filtering(target_frames, flows)

    if save_flow_video:
        video_writer = VideoWriter("flows.mp4", {"video_type": "mono", "fps": 30})
        video_writer(flows)

    if depth_maps is not None:
        depth_flow = depth.z_optical_flow_from_video(depth_maps)
        flows = depth.create_rgbd(flows, depth_flow)
    
    mode_shapes = motion.mode_shapes_from_optical_flow(flows, K, flows.shape[0]).squeeze(0)
    
    if mask is not None:
        mode_shapes = mask(mode_shapes, camera_pos=camera_pos)
        
    return mode_shapes.detach().cpu(), flows.detach().cpu()
    

def resize_spectrum_2_reference(
    motion_spectrum: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    reference_frame: np.ndarray
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Resizes the motion spectrum to match the size of the reference frame.

    Args:
        motion_spectrum (torch.Tensor | tuple[torch.Tensor, torch.Tensor]): The motion spectrum to be resized.
            If a tuple is provided, it is assumed to contain two tensors representing the motion spectrum in the x and y directions.
        reference_frame (np.ndarray): The reference frame used to determine the desired size of the motion spectrum.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]: The resized motion spectrum.
            If a tuple was provided as input, a tuple will be returned with the resized tensors for the x and y directions.
    """
    
    if isinstance(motion_spectrum, tuple):
        motion_spectrum = (
            torch.nn.functional.interpolate(motion_spectrum[0], size=(reference_frame.shape[0], reference_frame.shape[1]), mode="bilinear"),
            torch.nn.functional.interpolate(motion_spectrum[1], size=(reference_frame.shape[0], reference_frame.shape[1]), mode="bilinear")
        )
    else:
        motion_spectrum = torch.nn.functional.interpolate(motion_spectrum, size=(reference_frame.shape[0], reference_frame.shape[1]), mode="bilinear")
    
    return motion_spectrum


def calculate_modal_coordinate(
    mode_shape: torch.Tensor, 
    displacement: torch.Tensor,
    pixel: tuple[int, int],
    alpha: float = 1.0,
    maximize: str = "disp"
) -> torch.Tensor:
    """
    Calculate the modal coordinate from the mode shape and displacement.

    Args:
        mode_shape (torch.Tensor): The mode shape tensor.
        displacement (torch.Tensor): The displacement tensor.
        pixel (tuple[int, int]): The pixel coordinates.
        alpha (float, optional): The alpha value. Defaults to 1.0.
        maximize (str, optional): The maximize option. Defaults to "disp".

    Returns:
        torch.Tensor: The calculated modal coordinate tensor.
    """

    if len(mode_shape.shape) == 3:
        mode_shape = mode_shape.unsqueeze(0)
        
    # Calculate the magnitude of the vector
    magnitude = motion.calculate_modal_magnitude(mode_shape, displacement, pixel, alpha)
    # Calculate the phase of the vector
    phase = motion.calculate_modal_phase(mode_shape, displacement, pixel, maximize)
    # Convert the magnitude and phase to a complex tensor
    return math_helper.complex_from_magnitude_phase(magnitude, phase)
