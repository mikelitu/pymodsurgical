import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import PosixPath, Path

def calculate_norm(matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculate the norm of a complex tensor.
    """
    return torch.norm(matrix, p='fro')

def tensor_rotate_phase_to_real_axis(tensor: torch.Tensor) -> torch.Tensor:
    """
    Rotate the phase of a complex tensor to the real axis.
    """
    return torch.abs(tensor)


def calculate_relative_contribution(mode_i: torch.Tensor, mode_j: torch.Tensor) -> torch.Tensor:
    """
    Calculate the relative contribution of mode j to mode i.
    """
    norm_mode_i = calculate_norm(mode_i)
    norm_mode_j = calculate_norm(mode_j)

    return (norm_mode_i / norm_mode_j) * 100

def complex_from_magnitude_phase(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """
    Convert polar coordinates to a complex tensor.
    """
    real_part = magnitude * torch.cos(phase)
    imag_part = magnitude * torch.sin(phase)
    return real_part + 1j * imag_part


def complex_to_magnitude_phase(complex_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a complex tensor to polar coordinates.
    """
    magnitude = torch.abs(complex_tensor)
    phase = torch.angle(complex_tensor)
    return magnitude, phase


def get_conjugate(complex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Get the conjugate of a complex tensor.
    """
    return complex_tensor.conj()

def simplify_complex_tensor(complex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Simplify a complex tensor.
    """
    real_values = complex_tensor.real
    imag_values = complex_tensor.imag
    return torch.atan2(imag_values, real_values)


def mode_shape_2_complex(
    motion_spectrum: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a motion spectrum to a complex tensor.
    Args:
        motion_spectrum (torch.Tensor | tuple[torch.Tensor, torch.Tensor]): The motion spectrum.
        motion spectrum should be a tensor with shape (K, 2*dim, height, width) or a tuple of two tensors with shape (K, dim, height, width).
    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]: The complex motion spectrum.
        The shape of the new tensor is (K, dim, height, width) with real and complex components.
    """

    
    if isinstance(motion_spectrum, tuple):
        left_motion_spectrum = motion_spectrum[0]
        right_motion_spectrum = motion_spectrum[1]
        left_complex_motion_spectrum = mode_shape_2_complex(left_motion_spectrum)
        right_complex_motion_spectrum = mode_shape_2_complex(right_motion_spectrum)
        return left_complex_motion_spectrum, right_complex_motion_spectrum
    
    
    K, dim, height, width = motion_spectrum.shape
    if dim not in [4, 6]:
        raise ValueError("The dimension of the motion spectrum must be 4 for 2D tensors or 6 for 3D tensors.")
    
    complex_motion_spectrum = torch.zeros(K, 2 if dim==4 else 3, height, width).to(motion_spectrum.device, dtype=torch.cfloat)

    complex_motion_spectrum[:, 0, :, :] = motion_spectrum[:, 0, :, :] + 1j * motion_spectrum[:, 1, :, :]
    complex_motion_spectrum[:, 1, :, :] = motion_spectrum[:, 2, :, :] + 1j * motion_spectrum[:, 3, :, :]
    if dim == 6:
        complex_motion_spectrum[:, 2, :, :] = motion_spectrum[:, 4, :, :] + 1j * motion_spectrum[:, 5, :, :]
    
    return complex_motion_spectrum


def normalize_modal_coordinate(
    modal_coordinate: torch.Tensor
) -> torch.Tensor:
    """
    Normalize the modal coordinate.
    """
    norm = torch.linalg.norm(modal_coordinate, dim=1, ord=2)
    return modal_coordinate / norm.unsqueeze(-1)


def orthonormal_normalization(
    complex_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Compute the orthonormal normalization of a complex tensor.
    """
    shape = complex_tensor.shape
    shape_range = tuple(range(1, len(shape)))
    norm = torch.linalg.vector_norm(complex_tensor, dim=shape_range, keepdim=True)
    return complex_tensor / norm


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


def _norm_numpy(
    array: np.ndarray,
    as_img: bool = False
) -> np.ndarray:
    array = (array - array.min()) / (array.max() - array.min())
    if as_img:
        array = (255 * array).astype(np.uint8)
    return array


def _norm_torch(
    tensor: torch.Tensor,
    as_img: bool = False
) -> torch.Tensor:
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    if as_img:
        tensor = (255 * tensor).to(torch.uint8)
    return tensor   