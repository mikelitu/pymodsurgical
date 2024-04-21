import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import matplotlib.pyplot as plt
import numpy as np
from pathlib import PosixPath


def make_grid(img: torch.Tensor, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Create a grid of the same size as the input image.

    Args:
        img (torch.Tensor): Image tensor of shape (C, H, W) and dtype torch.float.

    Returns:
        grid (torch.Tensor): Grid tensor of shape (2, H, W) and dtype torch.float.
    """
    B, _, H, W = img.shape
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(device)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(device)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    return torch.cat((xx, yy), dim=1).float().to(img.device)


def warp_flow(
    img: torch.Tensor, 
    flow: torch.Tensor, 
    depth_map: np.ndarray,
    mask: np.ndarray | None = None, 
    near: float = 0.05, 
    far: float = 0.95,
    inverse: bool = False
) -> np.ndarray:
    
    """
    Warp an image using a flow field.

    Args:
        img (torch.Tensor): Image tensor of shape (C, H, W) and dtype torch.float.
        flow (torch.Tensor): Flow tensor of shape (2, H, W) and dtype torch.float.

    Returns:
        warped_img (torch.Tensor): Warped image tensor of shape (C, H, W) and dtype torch.float.
    """
    
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    if len(flow.shape) == 3:
        flow = flow.unsqueeze(0)
    
    depth_map = torch.from_numpy(depth_map).unsqueeze(0).to(flow.device)

    if mask is not None:
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(flow.device)

    _, _, H, W = img.shape
    grid = make_grid(img)
    grid = grid + flow

    # Apply the masking on the grid
    if mask is not None:
        grid = grid * mask

    # Perform depth culling based on near and far thresholds
    culled_depth_map = torch.where((depth_map >= near) &
                                   (depth_map <= far),
                                   depth_map, torch.zeros_like(depth_map))
    
    # Normalize culled depth map to range [0, 1]
    culled_depth_map = culled_depth_map / culled_depth_map.max()

    # Scale the grid to [-1, 1]
    grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / max(W - 1, 1) - 1.0
    grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / max(H - 1, 1) - 1.0
    grid = grid.permute(0, 2, 3, 1)
    img = img / 255.
    warped_img = torch.nn.functional.grid_sample(img, grid, mode="bicubic", padding_mode="zeros", align_corners=True)

    if inverse:
        warped_img = warped_img * (1 - culled_depth_map)
    else:
        warped_img = culled_depth_map * warped_img

    warped_img = warped_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    warped_img = np.clip(warped_img, 0., 1.)
    warped_img = (warped_img * 255).astype(np.uint8)
    return warped_img


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
    if motion_spectrum.ndim != 4 or (orig_shape[1] != 4 and orig_shape[1] != 6):
        raise ValueError("The motion spectrum should have shape (K, 4, H, W) or (K, 6, H, W)")
    
    motion_spectrum_X = torch.abs(motion_spectrum[:, 0] + 1j * motion_spectrum[:, 1])
    motion_spectrum_Y = torch.abs(motion_spectrum[:, 2] + 1j * motion_spectrum[:, 3])
    if orig_shape[1] == 6:
        motion_spectrum_Z = torch.abs(motion_spectrum[:, 4] + 1j * motion_spectrum[:, 5])
        motion_spectrum = torch.stack([motion_spectrum_X, motion_spectrum_Y, motion_spectrum_Z], dim=1)
    else:
        motion_spectrum = torch.stack([motion_spectrum_X, motion_spectrum_Y], dim=1)
    max_norm = torch.sum(motion_spectrum**2, dim=1).sqrt().max()
    epsilon = torch.finfo(motion_spectrum.dtype).eps
    motion_spectrum = motion_spectrum / (max_norm + epsilon)
    # img = _normalized_motion_spectrum_to_image(motion_spectrum)
    if orig_shape[1] == 6:
        return motion_spectrum[:, 0], motion_spectrum[:, 1], motion_spectrum[:, 2]
    
    return motion_spectrum[:, 0], motion_spectrum[:, 1]


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
    """
    Estimate the optical flow between two frames.

    Args:
        model (torch.nn.Module): The flow estimator model.
        reference_frames (torch.Tensor): The reference frames.
        target_sequences (torch.Tensor): The target sequences.

    Returns:
        torch.Tensor: The estimated optical flow.

    """
    batch_size = reference_frames.shape[0]
    
    flows = torch.zeros((batch_size, 2, reference_frames.shape[2], reference_frames.shape[3])).to(reference_frames.device)
    
    with torch.no_grad():
        reference_frame = reference_frames
        target_sequence = target_sequences
        # Upscale the image to the minimum required size for the flow estimator
        if reference_frame.shape[2] < 128 or reference_frame.shape[3] < 128:
            in_height, in_width = reference_frame.shape[2], reference_frame.shape[3]
            reference_frame = torch.nn.functional.interpolate(reference_frame, size=(128, 128), mode="bilinear")
            target_sequence = torch.nn.functional.interpolate(target_sequence, size=(128, 128), mode="bilinear")
        else:
            in_height, in_width = reference_frame.shape[2], reference_frame.shape[3]

        
        list_of_flows = model(reference_frame, target_sequence)
        
        cur_flows = list_of_flows[-1]
        if in_height < 128 or in_width < 128:
            flows = torch.nn.functional.interpolate(cur_flows, size=(in_height, in_width), mode="bilinear")
        else:
            flows = cur_flows

    return flows


def motion_texture_from_flow_field(
    flow_field: torch.Tensor, 
    K: int,
    timestep: int
) -> torch.Tensor:
    
    if len(flow_field.shape) == 4:
        flow_field = flow_field.view(-1, timestep, flow_field.shape[1], flow_field.shape[2], flow_field.shape[3])

    batch_size, _, dim, height, width = flow_field.shape
    if dim not in [2, 3]:
        raise ValueError("The dimension of the flow field must be 2 for 2D tensors or 3 for 3D tensors.")
    
    motion_texture = torch.zeros((batch_size, K, 4 if dim==2 else 6, height, width)).to(flow_field.device)
    for i in range(height):
        for j in range(width):
            x_t = flow_field[:, :, 0, i, j]
            y_t = flow_field[:, :, 1, i, j]
            Y_f = torch.fft.rfft(x_t, timestep, dim=-1)
            X_f = torch.fft.rfft(y_t, timestep, dim=-1)
            motion_texture[:, :, 0, i, j] = (1.0 / K) * X_f[:, 1:K+1].real
            motion_texture[:, :, 1, i, j] = (1.0 / K) * X_f[:, 1:K+1].imag
            motion_texture[:, :, 2, i, j] = (1.0 / K) * Y_f[:, 1:K+1].real
            motion_texture[:, :, 3, i, j] = (1.0 / K) * Y_f[:, 1:K+1].imag
            if dim == 3:
                z_t = flow_field[:, :, 2, i, j]
                Z_f = torch.fft.rfft(z_t, timestep, dim=-1)
                motion_texture[:, :, 4, i, j] = (1.0 / K) * Z_f[:, 1:K+1].real
                motion_texture[:, :, 5, i, j] = (1.0 / K) * Z_f[:, 1:K+1].imag

    return motion_texture


def flow_field_from_motion_texture(
    motion_texture: torch.Tensor,
    timesteps: int,
    K: int
) -> torch.Tensor:
    
    if len(motion_texture.shape) == 4:
        motion_texture = motion_texture.view(-1, K, motion_texture.shape[1], motion_texture.shape[2], motion_texture.shape[3])

    batch_size, _, dim, height, width = motion_texture.shape
    if dim not in [4, 6]:
        raise ValueError("The dimension of the motion texture must be 4 for 2D tensors or 6 for 3D tensors.")
    
    flow_field = torch.zeros((batch_size, timesteps, 2 if dim==4 else 3, height, width)).to(motion_texture.device)
    for i in range(height):
        for j in range(width):
            X_f = motion_texture[:, :, 0, i, j].float() * K + 1j * motion_texture[:, :, 1, i, j].float() * K
            Y_f = motion_texture[:, :, 2, i, j].float() * K + 1j * motion_texture[:, :, 3, i, j].float() * K
            x_t = torch.fft.irfft(X_f, timesteps, dim=-1)
            y_t = torch.fft.irfft(Y_f, timesteps, dim=-1)
            flow_field[:, :, 0, i, j] = x_t
            flow_field[:, :, 1, i, j] = y_t
            if dim==6:
                Z_f = motion_texture[:, :, 4, i, j].float() * K + 1j * motion_texture[:, :, 5, i, j].float() * K
                z_t = torch.fft.irfft(Z_f, timesteps, dim=-1)
                flow_field[:, :, 2, i, j] = z_t

    return flow_field


def get_motion_frequencies(
    timesteps: int,
    K: int,
    sampling_period: float
) -> torch.Tensor:
    """
    Compute the motion frequencies.

    Args:
        timesteps (int): Number of timesteps.
        K (int): Number of frequencies.
        sample_rate (float): The sample rate.

    Returns:
        frequencies (torch.Tensor): Tensor of shape (K,) and dtype torch.float.
    """

    frequency_spacing = 1.0 / (timesteps * sampling_period)
    frequencies = torch.fft.fftfreq(timesteps, d=sampling_period)[1:K+1]
    return 2 * torch.pi * (frequencies * frequency_spacing)