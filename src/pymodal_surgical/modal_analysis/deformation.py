import torch
from . import functions


def normalize_deformation_map(displacement_map: torch.Tensor) -> torch.Tensor:
    return (displacement_map - displacement_map.min()) / (displacement_map.max() - displacement_map.min())


def calculate_deformation_map_from_modal_coordinate(
    mode_shape: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    modal_coordinate: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the deformation map from the mode shape and modal coordinate.
    """

    if isinstance(mode_shape, tuple):
        left_mode_shape = mode_shape[0]
        right_mode_shape = mode_shape[1]
        left_deformation_map = calculate_deformation_map_from_modal_coordinate(left_mode_shape, modal_coordinate[0])
        right_deformation_map = calculate_deformation_map_from_modal_coordinate(right_mode_shape, modal_coordinate[1])
        return left_deformation_map, right_deformation_map
    
    modal_coordinate = modal_coordinate.unsqueeze(-1).unsqueeze(-1)
    deformation_map = (mode_shape * modal_coordinate).real.sum(dim=0)
    return deformation_map 


def calculate_deformation_map_from_displacement(
    mode_shape: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    displacement: torch.Tensor,
    pixel: tuple[int, int],
    alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    
    """
    Calculate the deformation map from the mode shape and displacement.
    """

    # Calculate the modal coordinate and reshape for multiplication
    modal_coordinates = functions.calculate_modal_coordinate(mode_shape, displacement, pixel, alpha)
    modal_coordinates = modal_coordinates.unsqueeze(-1).unsqueeze(-1)
    # modal_coordinates = normalize_modal_coordinate(modal_coordinates)

    # Calculate the deformation map by weighting the mode shape by the modal coordinate
    deformation_maps = (mode_shape * modal_coordinates).real.sum(dim=0)
    return deformation_maps, modal_coordinates



