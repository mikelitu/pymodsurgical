from pymodal_surgical import optical_flow, depth, utils
from pathlib import PosixPath
import torch
import numpy as np
from pymodal_surgical.filtering import GaussianFiltering
from pymodal_surgical.masking import Masking
from pymodal_surgical.video_writer import VideoWriter

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = optical_flow.load_flow_model(device)

def calculate_motion_spectrum(
    frames: list[np.ndarray], 
    K: int,
    depth_maps: list[np.ndarray] | None = None,
    batch_size: int = 500, 
    filtered: bool = True,
    mask: Masking | None = None,
    camera_pos: str | None = None,
    save_flow_video: bool = False
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
    
    motion_spectrum = optical_flow.motion_texture_from_flow_field(flows, K, flows.shape[0]).squeeze(0)
    
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
                
            tmp_save_dir = utils.create_save_dir(save_dir, filename)
            print(f"Saving motion spectrum to: {tmp_save_dir}")
            dim = motion_spectrum[i].shape[1]
            if dim == 4:
                img_motion_spectrum_X, img_motion_spectrum_Y = optical_flow.motion_spectrum_2_grayimage(motion_spectrum[i])
                optical_flow.plot_and_save([img_motion_spectrum_X, img_motion_spectrum_Y], tmp_save_dir, cmap="plasma")
            else:
                img_motion_spectrum_X, img_motion_spectrum_Y, img_motion_spectrum_Z = optical_flow.motion_spectrum_2_grayimage(motion_spectrum[i])
                optical_flow.plot_and_save([img_motion_spectrum_X, img_motion_spectrum_Y, img_motion_spectrum_Z], tmp_save_dir, cmap="plasma")
    else:
        filename = "motion_spectrum.png"
        if filtered:
            filename = "filtered_" + filename
        if masked:
            filename = "masked_" + filename
        save_dir = utils.create_save_dir(save_dir, filename)
        print("Saving motion spectrum to: ", save_dir)
        dim = motion_spectrum.shape[1]
        if dim == 4:
            img_motion_spectrum_X, img_motion_spectrum_Y = optical_flow.motion_spectrum_2_grayimage(motion_spectrum)
            optical_flow.plot_and_save([img_motion_spectrum_X, img_motion_spectrum_Y], save_dir, cmap="plasma")
        else:
            img_motion_spectrum_X, img_motion_spectrum_Y, img_motion_spectrum_Z = optical_flow.motion_spectrum_2_grayimage(motion_spectrum)
            optical_flow.plot_and_save([img_motion_spectrum_X, img_motion_spectrum_Y, img_motion_spectrum_Z], save_dir, cmap="plasma")
    

def resize_spectrum_2_reference(
    motion_spectrum: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    reference_frame: np.ndarray
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    
    if isinstance(motion_spectrum, tuple):
        motion_spectrum = (torch.nn.functional.interpolate(motion_spectrum[0], size=(reference_frame.shape[0], reference_frame.shape[1]), mode="bilinear"), torch.nn.functional.interpolate(motion_spectrum[1], size=(reference_frame.shape[0], reference_frame.shape[1]), mode="bilinear"))
    else:
        motion_spectrum = torch.nn.functional.interpolate(motion_spectrum, size=(reference_frame.shape[0], reference_frame.shape[1]), mode="bilinear")
    
    return motion_spectrum