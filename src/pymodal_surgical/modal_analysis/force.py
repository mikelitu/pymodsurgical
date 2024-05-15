import torch
from . import math_helper


def calculate_motion_compensation_matrix(
    frequencies: torch.Tensor,
    timestep: float = 0.1,
    alpha: float = 0.1,
    beta: float = 0.1
) -> torch.Tensor:
    """
    Calculates the motion compensation matrix for a given set of frequencies.

    Args:
        frequencies (torch.Tensor): Tensor containing the frequencies.
        timestep (float, optional): Time step value. Defaults to 0.1.
        alpha (float, optional): Alpha value. Defaults to 0.1.
        beta (float, optional): Beta value. Defaults to 0.1.

    Returns:
        torch.Tensor: Tensor containing the diagonal of the motion matrix.
    """
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
    pixel: tuple[tuple[int, int], tuple[int, int]] | tuple[int, int],
    simplify_force: bool = True,
    timestep: float = 0.1,
    alpha: float = 0.1,
    beta: float = 0.1
) -> torch.Tensor:
    """
    Calculate the force from the optical flow and the previous displacement vector.

    Args:
        mode_shape (torch.Tensor): The mode shape tensor.
        displacement_map (torch.Tensor): The displacement map tensor.
        modal_coordinate (torch.Tensor): The modal coordinate tensor.
        frequencies (torch.Tensor): The frequencies tensor.
        pixel (tuple[int, int]): The pixel coordinates.
        timestep (float, optional): The timestep value. Defaults to 0.1.
        alpha (float, optional): The alpha value. Defaults to 0.1.
        beta (float, optional): The beta value. Defaults to 0.1.

    Returns:
        torch.Tensor: The calculated force tensor.
    """
    
    if isinstance(pixel[0], tuple):
        pix_h = pixel[0]
        pix_w = pixel[1]
    
    area = (pix_w[1] - pix_w[0]) * (pix_h[1] - pix_h[0])
    # Calculate the motion compensation matrix
    S = calculate_motion_compensation_matrix(frequencies, timestep, alpha, beta)
    modal_coordinate = modal_coordinate.reshape(-1, 2, 1)
    
    displacement_vector = displacement_map[:, pix_h[0]:pix_h[1], pix_w[0]:pix_w[1]]

    force_sign = torch.sign(displacement_vector).squeeze(-1)
    pixel_mode_shape = mode_shape[:, :, pix_h[0]:pix_h[1], pix_w[0]:pix_w[1]]
    trans_pixel_mode_shape = pixel_mode_shape.transpose(2, 3)
    mode_shape_mult = pixel_mode_shape * S.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) @ trans_pixel_mode_shape
    inv_mode_shape_mult = mode_shape_mult.pinverse()
    inv_mode_shape_mult = math_helper.orthonormal_normalization(inv_mode_shape_mult)
    
    if torch.all(modal_coordinate == 0):
        state_difference = displacement_vector
    else:
        state_difference = displacement_vector - pixel_mode_shape * modal_coordinate

    # Apply the constraint problem to calculate the force from a manipulation of the displacement vector

    force = (2 / (timestep ** 2)) * (inv_mode_shape_mult @ state_difference.to(dtype=torch.cfloat))
    
    # Force is the sum of the absolute value of the force in the x and y direction
    
    force = force.abs().sum(dim=0)
    force = force_sign * force
    
    if simplify_force:
        force = force.mean(dim=1).mean(dim=1)

    return force