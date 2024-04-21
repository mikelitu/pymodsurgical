import torch
from . import complex

def normalize_displacement_map(displacement_map: torch.Tensor) -> torch.Tensor:
    return (displacement_map - displacement_map.min()) / (displacement_map.max() - displacement_map.min())

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
    magnitude = calculate_modal_magnitude(mode_shape, displacement, pixel, alpha)
    # Calculate the phase of the vector
    phase = calculate_modal_phase(mode_shape, displacement, pixel, maximize)
    # Convert the magnitude and phase to a complex tensor
    return complex.complex_from_magnitude_phase(magnitude, phase)


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
    modal_coordinates = calculate_modal_coordinate(mode_shape, displacement, pixel, alpha)
    modal_coordinates = modal_coordinates.unsqueeze(-1).unsqueeze(-1)
    # modal_coordinates = normalize_modal_coordinate(modal_coordinates)

    # Calculate the deformation map by weighting the mode shape by the modal coordinate
    deformation_maps = (mode_shape * modal_coordinates).real.sum(dim=0)
    return deformation_maps, modal_coordinates


def calculate_motion_compensation_matrix(
    frequencies: torch.Tensor,
    timestep: float = 0.1,
    alpha: float = 0.1,
    beta: float = 0.1
):
    # conjugate_frequencies = get_conjugate(frequencies)
    numerator = torch.exp(timestep * frequencies)
    alpha_beta = alpha * frequencies + beta

    # Secure the denominator to be non negative by taking the absolute value before the square root
    denominator = torch.sqrt(torch.abs(alpha_beta ** 2 - 4 * frequencies))

    # Calculate the diagonal of the motion matrix
    diagonal = numerator / (denominator + torch.finfo(torch.float32).eps)
    return diagonal

def calculate_force_from_displacement_map(
    mode_shape: torch.Tensor,
    displacement_map: torch.Tensor,
    modal_coordinate: torch.Tensor,
    frequencies: torch.Tensor,
    pixel: tuple[int, int],
    timestep: float = 0.1,
    alpha: float = 0.1,
    beta: float = 0.1
) -> torch.Tensor:
    
    """
    Calculate the force from the optical flow and the previous displacement vector.
    """
    # Calculate the motion compensation matrix
    S = calculate_motion_compensation_matrix(frequencies, timestep, alpha, beta)
    modal_coordinate = modal_coordinate.reshape(-1, 2, 1)

    displacement_vector = displacement_map[:, *pixel].unsqueeze(-1)
    force_sign = torch.sign(displacement_vector).squeeze(-1)
    pixel_mode_shape = mode_shape[:, :, *pixel].unsqueeze(-1)
    trans_pixel_mode_shape = pixel_mode_shape.transpose(1, 2)
    mode_shape_mult = pixel_mode_shape * S.unsqueeze(-1).unsqueeze(-1) @ trans_pixel_mode_shape
    inv_mode_shape_mult = mode_shape_mult.pinverse()
    inv_mode_shape_mult = complex.orthonormal_normalization(inv_mode_shape_mult)
    
    state_difference = displacement_vector - pixel_mode_shape * modal_coordinate

    # Apply the constraint problem to calculate the force from a manipulation of the displacement vector
    force = (2 / (timestep ** 2)) * (inv_mode_shape_mult * state_difference)
    
    # Force is the sum of the absolute value of the force in the x and y direction
    force = force.abs().sum(dim=0).sum(dim=1)
    force = force_sign * force

    return force / 1.e6
