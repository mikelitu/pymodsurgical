from . import depth, optical_flow, modal_analysis, math_helper
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
    batch_size: int = 500, 
    filtered: bool = True,
    mask: Masking | None = None,
    camera_pos: str | None = None,
    save_flow_video: bool = False,
) -> torch.Tensor:
    
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
            flows[start:end] = optical_flow.estimate_flow(model, reference_frames, target_frames[start:end]).squeeze(0)
    else:
        flows = optical_flow.estimate_flow(model, reference_frames, target_frames).squeeze(0)

    if filtered:
        filtering = GaussianFiltering((11, 11), 3.0)
        flows = filtering(target_frames, flows)

    if save_flow_video:
        video_writer = VideoWriter("flows.mp4", {"video_type": "mono", "fps": 30})
        video_writer(flows)

    if depth_maps is not None:
        depth_flow = depth.z_optical_flow_from_video(depth_maps)
        flows = depth.create_rgbd(flows, depth_flow)
    
    motion_spectrum = modal_analysis.mode_shapes_from_optical_flow(flows, K, flows.shape[0]).squeeze(0)
    
    if mask is not None:
        motion_spectrum = mask(motion_spectrum, camera_pos=camera_pos)
        
    return motion_spectrum.detach().cpu()

    

def resize_spectrum_2_reference(
    motion_spectrum: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    reference_frame: np.ndarray
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    
    if isinstance(motion_spectrum, tuple):
        motion_spectrum = (torch.nn.functional.interpolate(motion_spectrum[0], size=(reference_frame.shape[0], reference_frame.shape[1]), mode="bilinear"), torch.nn.functional.interpolate(motion_spectrum[1], size=(reference_frame.shape[0], reference_frame.shape[1]), mode="bilinear"))
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
    """

    if len(mode_shape.shape) == 3:
        mode_shape = mode_shape.unsqueeze(0)
        
    # Calculate the magnitude of the vector
    magnitude = modal_analysis.calculate_modal_magnitude(mode_shape, displacement, pixel, alpha)
    # Calculate the phase of the vector
    phase = modal_analysis.calculate_modal_phase(mode_shape, displacement, pixel, maximize)
    # Convert the magnitude and phase to a complex tensor
    return math_helper.complex_from_magnitude_phase(magnitude, phase)
