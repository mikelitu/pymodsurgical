import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image, _make_colorwheel
import matplotlib.pyplot as plt
import numpy as np
from pathlib import PosixPath


def plot_and_save(imgs: list[torch.Tensor] | torch.Tensor, filename: PosixPath | None = None, **imshow_kwargs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    
    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 5, num_rows * 5), squeeze=False)

    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx][col_idx]
            img = torchvision.transforms.functional.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def motion_spectrum_2_grayimage(motion_spectrum: torch.Tensor) -> torch.Tensor:
    """
    Converts a motion spectrum to a RGB image.

    Args:
        motion_texture (torch.Tensor): Spectrum of shape (K, 4, H, W) and dtype torch.float.

    Returns:
        img (Tensor): Image Tensor of dtype uint8 and shape where each color corresponds to 
        a given spectrum direction. Shape is (K, 3, H, W).
    """

    if motion_spectrum.dtype != torch.float:
        raise ValueError("The motion spectrum should have dtype torch.float")
    
    orig_shape = motion_spectrum.shape
    if motion_spectrum.ndim != 4 and orig_shape[1] != 4:
        raise ValueError("The motion spectrum should have shape (K, 4, H, W)")
    
    motion_spectrum_X = torch.abs(motion_spectrum[:, 0] + 1j * motion_spectrum[:, 1])
    motion_spectrum_Y = torch.abs(motion_spectrum[:, 2] + 1j * motion_spectrum[:, 3])
    motion_spectrum = torch.stack([motion_spectrum_X, motion_spectrum_Y], dim=1)
    max_norm = torch.sum(motion_spectrum**2, dim=1).sqrt().max()
    epsilon = torch.finfo(motion_spectrum.dtype).eps
    motion_spectrum = motion_spectrum / (max_norm + epsilon)
    # img = _normalized_motion_spectrum_to_image(motion_spectrum)
    return motion_spectrum[:, 0], motion_spectrum[:, 1]


def _normalized_motion_spectrum_to_image(motion_spectrum: torch.Tensor) -> torch.Tensor:

    N, _, H, W = motion_spectrum.shape
    device = motion_spectrum.device
    motion_spectrum_img = torch.zeros((N, 3, H, W), dtype=torch.uint8).to(device)
    colorwheel = _make_colorwheel().to(device) # (55, 3)
    num_cols = colorwheel.shape[0] # 55
    norm = torch.sum(motion_spectrum**2, dim=1).sqrt()
    a = torch.atan2(-motion_spectrum[:, 1], -motion_spectrum[:, 0]) / torch.pi
    fk = (a + 1) / 2 * (num_cols - 1)
    k0 = torch.floor(fk).to(torch.long)
    k1 = k0 + 1
    k1[k1 == num_cols] = 0
    f = fk - k0

    for c in range(colorwheel.shape[1]):
        tmp = colorwheel[:, c]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        col = 1 - norm * (1 - col)
        motion_spectrum_img[:, c] = torch.floor(255 * col)
    
    return motion_spectrum_img


def load_flow_model(device: torch.device = torch.device("cuda:0")) -> torch.nn.Module:
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
    model.to(device)
    model.eval()
    return model


def preprocess_for_raft(batch: torch.Tensor):
    tranforms = T.Compose(
        [
            T.ToTensor(),
            T.Resize((128, 128)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # map [0, 1] into [-1, 1]
        ]
    )
    return tranforms(batch)


def estimate_flow(
    model: torch.nn.Module,
    reference_frames: torch.Tensor,
    target_sequences: torch.Tensor,
) -> torch.Tensor:
    
    # Check that the target images have the same shape as the reference frame
    batch_size = reference_frames.shape[0]
    
    if len(target_sequences.shape) == 4:
        target_sequences = target_sequences.view(batch_size, -1, 3, target_sequences.shape[2], target_sequences.shape[3])
    
    flows = torch.zeros((batch_size, target_sequences.shape[1], 2, reference_frames.shape[2], reference_frames.shape[3])).to(reference_frames.device)
    
    with torch.no_grad():
        for i in range(batch_size):
            reference_frame = reference_frames[i].unsqueeze(0)
            target_sequence = target_sequences[i]
            # Upscale the image to the minimum required size for the flow estimator
            if reference_frame.shape[2] < 128 or reference_frame.shape[3] < 128:
                in_height, in_width = reference_frame.shape[2], reference_frame.shape[3]
                reference_frame = torch.nn.functional.interpolate(reference_frame, size=(128, 128), mode="bilinear")
                target_sequence = torch.nn.functional.interpolate(target_sequence, size=(128, 128), mode="bilinear")
            else:
                in_height, in_width = reference_frame.shape[2], reference_frame.shape[3]

            # Extend the reference frame batch size to match the length of the target sequence
            reference_frame = reference_frame.repeat(len(target_sequence), 1, 1, 1)

            
            list_of_flows = model(reference_frame, target_sequence)
            
            cur_flows = list_of_flows[-1]
            if in_height < 128 or in_width < 128:
                flows[i] = torch.nn.functional.interpolate(cur_flows, size=(in_height, in_width), mode="bilinear")
            else:
                flows[i] = cur_flows

    return flows


def motion_texture_from_flow_field(
    flow_field: torch.Tensor, 
    K: int,
    timestep: int
) -> torch.Tensor:
    
    if len(flow_field.shape) == 4:
        flow_field = flow_field.view(-1, timestep, flow_field.shape[1], flow_field.shape[2], flow_field.shape[3])

    batch_size, _, _, height, width = flow_field.shape
    motion_texture = torch.zeros((batch_size, K, 4, height, width)).to(flow_field.device)
    for i in range(height):
        for j in range(width):
            x_t = flow_field[:, :, 0, i, j]
            y_t = flow_field[:, :, 1, i, j]
            Y_f = torch.fft.rfft(x_t, timestep, dim=-1)
            X_f = torch.fft.rfft(y_t, timestep, dim=-1)
            motion_texture[:, :, 0, i, j] = (1.0 / K) * X_f[:, :K].real
            motion_texture[:, :, 1, i, j] = (1.0 / K) * X_f[:, :K].imag
            motion_texture[:, :, 2, i, j] = (1.0 / K) * Y_f[:, :K].real
            motion_texture[:, :, 3, i, j] = (1.0 / K) * Y_f[:, :K].imag

    return motion_texture


def flow_field_from_motion_texture(
    motion_texture: torch.Tensor,
    timesteps: int,
    K: int
) -> torch.Tensor:
    
    if len(motion_texture.shape) == 4:
        motion_texture = motion_texture.view(-1, K, motion_texture.shape[1], motion_texture.shape[2], motion_texture.shape[3])

    batch_size, _, _, height, width = motion_texture.shape
    flow_field = torch.zeros((batch_size, timesteps, 2, height, width)).to(motion_texture.device)
    for i in range(height):
        for j in range(width):
            X_f = motion_texture[:, :, 0, i, j].float() * K + 1j * motion_texture[:, :, 1, i, j].float() * K
            Y_f = motion_texture[:, :, 2, i, j].float() * K + 1j * motion_texture[:, :, 3, i, j].float() * K
            x_t = torch.fft.irfft(X_f, timesteps, dim=-1)
            y_t = torch.fft.irfft(Y_f, timesteps, dim=-1)
            flow_field[:, :, 0, i, j] = x_t
            flow_field[:, :, 1, i, j] = y_t

    return flow_field
