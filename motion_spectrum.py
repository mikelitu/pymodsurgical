from optical_flow import estimate_flow, load_flow_model, preprocess_for_raft, plot_and_save, motion_spectrum_2_grayimage, motion_texture_from_flow_field
from pathlib import PosixPath
import torch
import numpy as np
from utils import create_save_dir
from filtering import GaussianFiltering
from masking import Masking
from video_writer import VideoWriter

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = load_flow_model(device)

def calculate_motion_spectrum(
    frames: list[np.ndarray], 
    K: int, 
    batch_size: int = 500, 
    filtered: bool = True,
    mask: Masking | None = None,
    camera_pos: str | None = None,
    save_flow_video: bool = False
) -> torch.Tensor:
    
    preprocess_frames = [preprocess_for_raft(frame) for frame in frames]
    preprocess_frames = torch.stack(preprocess_frames).to(device)
    
    B = preprocess_frames.shape[0]
    flows = torch.zeros((B - 1, 2, preprocess_frames.shape[2], preprocess_frames.shape[3])).to(device)

    reference_frame = preprocess_frames[0].unsqueeze(0)

    if B > batch_size:
        number_of_batches = B // batch_size
        for i in range(number_of_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            flows[start:end] = estimate_flow(model, reference_frame, preprocess_frames[start:end]).squeeze(0)
    else:
        flows = estimate_flow(model, reference_frame, preprocess_frames[1:]).squeeze(0)

    if filtered:
        filtering = GaussianFiltering((3, 3), 0.5)
        flows = filtering(preprocess_frames[1:], flows)

    if save_flow_video:
        video_writer = VideoWriter("flows.mp4", {"fps": 30}, "mono")
        video_writer(flows)

    motion_spectrum = motion_texture_from_flow_field(flows, K, flows.shape[0]).squeeze(0)
    
    if mask is not None:
        motion_spectrum = mask(motion_spectrum, camera_pos=camera_pos)
        
    return motion_spectrum.detach().cpu()


def save_motion_spectrum(
    motion_spectrum: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    save_dir: PosixPath | str,
    filtered: bool = False,
    masked: bool = False
) -> None:
    
    if isinstance(motion_spectrum, tuple):
        filenames = {0: "left_motion_spectrum.png", 1: "right_motion_spectrum.png"}
        for i in range(len(motion_spectrum)):
            filename = filenames[i]
            if filtered:
                filename = "filtered_" + filename
            if masked:
                filename = "masked_" + filename
                
            tmp_save_dir = create_save_dir(save_dir, filename)
            print(f"Saving motion spectrum to: {tmp_save_dir}")
            img_motion_spectrum_X, img_motion_spectrum_Y = motion_spectrum_2_grayimage(motion_spectrum[i])
            plot_and_save([img_motion_spectrum_X, img_motion_spectrum_Y], tmp_save_dir, cmap="plasma")
    else:
        filename = "motion_spectrum.png"
        if filtered:
            filename = "filtered_" + filename
        if masked:
            filename = "masked_" + filename
        save_dir = create_save_dir(save_dir, filename)
        print("Saving motion spectrum to: ", save_dir)
        img_motion_spectrum_X, img_motion_spectrum_Y = motion_spectrum_2_grayimage(motion_spectrum)
        plot_and_save([img_motion_spectrum_X, img_motion_spectrum_Y], save_dir, cmap="plasma")