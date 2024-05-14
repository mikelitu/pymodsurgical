import torch
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import numpy as np
from . import math_helper


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
    grid = math_helper.make_grid(img)
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


def load_flow_model(device: torch.device = torch.device("cuda:0")) -> torch.nn.Module:
    """
    Loads the optical flow model.

    Args:
        device (torch.device, optional): The device to load the model on. Defaults to "cuda:0".

    Returns:
        torch.nn.Module: The loaded optical flow model.
    """
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
    model.to(device)
    model.eval()
    return model


def preprocess_for_raft(batch: torch.Tensor):
    """
    Preprocesses the input batch of images for the RAFT model.

    Args:
        batch (torch.Tensor): The input batch of images.

    Returns:
        torch.Tensor: The preprocessed batch of images.
    """
    tranforms = T.Compose(
        [
            T.ToTensor(),
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
        if reference_frame.shape[2] % 8 != 0 or reference_frame.shape[3] % 8 != 0:
            in_height, in_width = reference_frame.shape[2], reference_frame.shape[3]
            reference_frame = torch.nn.functional.interpolate(reference_frame, size=(128, 128), mode="bilinear")
            target_sequence = torch.nn.functional.interpolate(target_sequence, size=(128, 128), mode="bilinear")
        else:
            in_height, in_width = reference_frame.shape[2], reference_frame.shape[3]

        
        list_of_flows = model(reference_frame, target_sequence)
        
        cur_flows = list_of_flows[-1]
        if in_height != reference_frame.shape[2] or in_width != reference_frame.shape[3]:
            flows = torch.nn.functional.interpolate(cur_flows, size=(in_height, in_width), mode="bilinear")
        else:
            flows = cur_flows

    return flows


