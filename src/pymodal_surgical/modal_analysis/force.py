import torch
from . import math_helper


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
    inv_mode_shape_mult = math_helper.orthonormal_normalization(inv_mode_shape_mult)
    
    state_difference = displacement_vector - pixel_mode_shape * modal_coordinate

    # Apply the constraint problem to calculate the force from a manipulation of the displacement vector
    force = (2 / (timestep ** 2)) * (inv_mode_shape_mult * state_difference)
    
    # Force is the sum of the absolute value of the force in the x and y direction
    force = force.abs().sum(dim=0).sum(dim=1)
    force = force_sign * force

    return force / 1.e6