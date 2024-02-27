import torch
import numpy as np
from complex import normalize_modal_coordinate, complex_from_magnitude_phase

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

    # Calculate the magnitude of the vector
    magnitude = calculate_modal_magnitude(mode_shape, displacement, pixel, alpha)
    # Calculate the phase of the vector
    phase = calculate_modal_phase(mode_shape, displacement, pixel, maximize)
    # Convert the magnitude and phase to a complex tensor
    return complex_from_magnitude_phase(magnitude, phase)

def calculate_modal_magnitude(
    mode_shape: torch.Tensor, 
    displacement: torch.Tensor,
    pixel: tuple[int, int],
    alpha: float = 1.0
) -> torch.Tensor:
    """
    Calculate the modal coordinate magnitude from the mode shape and displacement.
    """

    # Calculate the modal coordinate
    displacement = displacement.unsqueeze(1).T
    displacement = displacement.repeat(mode_shape.shape[0], 1)
    mode_pixel = mode_shape[:, :, *pixel]
    batch_mm = (displacement * mode_pixel).abs()
    return batch_mm * alpha

def calculate_modal_phase(
    mode_shape: torch.Tensor, 
    displacement: torch.Tensor,
    pixel: tuple[int, int],
    maximize: str = "disp"
) -> torch.Tensor:
    """
    Calculate the modal coordinate from the mode shape and displacement.
    """
    if maximize not in ["disp", "velocity"]:
        raise ValueError("maximize must be 'disp' or 'velocity'")
    # Get the displacement at the correct shape
    displacement = displacement.unsqueeze(1).T
    displacement = displacement.repeat(mode_shape.shape[0], 1)
    # Get the mode shape at the correct pixel
    mode_pixel = mode_shape[:, :, *pixel]
    # Get the phase of the dot product
    angles = torch.angle(displacement * mode_pixel)
    if maximize == "disp":
        return -angles
    else:
        return -angles + torch.pi/2
    

def calculate_deformation_map(
    mode_shape: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    displacement: torch.Tensor,
    pixel: tuple[int, int],
    alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the deformation map from the mode shape and displacement.
    """

    # Calculate the modal coordinate and reshape for multiplication
    modal_coordinates = calculate_modal_coordinate(mode_shape, displacement, pixel, alpha)
    modal_coordinates = modal_coordinates.unsqueeze(-1).unsqueeze(-1)
    # Normalize the modal coordinate
    # normalized_modal_coordinates = normalize_modal_coordinate(modal_coordinates)
    # Calculate the deformation map by weighting the mode shape by the modal coordinate
    deformation_maps = (mode_shape * modal_coordinates).abs().sum(dim=0)
    # norm_deformation_maps = (deformation_maps - deformation_maps.min()) / (deformation_maps.max() - deformation_maps.min())
    return deformation_maps, modal_coordinates
