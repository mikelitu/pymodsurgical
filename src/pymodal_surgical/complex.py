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


def motion_spectrum_2_complex(
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
        left_complex_motion_spectrum = motion_spectrum_2_complex(left_motion_spectrum)
        right_complex_motion_spectrum = motion_spectrum_2_complex(right_motion_spectrum)
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


def plot_modal_coordinates(
    modal_coordinates: torch.Tensor | np.ndarray,
    displacement: torch.Tensor | None = None,
    pixel: tuple[int, int] | None = None,
    show: bool = True,
    save: bool = False,
    save_dir: str | PosixPath = "./figures"
) -> None:
    """
    Plot the modal coordinates.
    """

    # Transform the modal coordinates to numpy
    if isinstance(modal_coordinates, torch.Tensor):
        modal_coordinates = normalize_modal_coordinate(modal_coordinates)
        modal_coordinates = modal_coordinates.detach().cpu().numpy()
    
    plot_names = ["X", "Y", "Z"]

    # Create the figure
    fig, axs = plt.subplots(ncols=modal_coordinates.shape[1], figsize=(16, 7))
    
    # Setting the properties of the plot
    for i, ax in enumerate(axs):
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True)
        ax.axhline(0, color='black', lw=0.5)
        ax.axvline(0, color='black', lw=0.5)
        ax.set_title(f'Modal Coordinates {plot_names[i]}')
    
    for mc in modal_coordinates:
        ax.set_aspect('equal', 'box')

        # Plot each modal coordinate
        for i, ax in enumerate(axs):
            ax.quiver(0, 0, mc[i].real, mc[i].imag, angles='xy', scale_units='xy', scale=1, color='r')

    # Set the title of the plot
    if modal_coordinates.shape[1] == 2:
        str_displacement = f"({displacement[0]}_{displacement[1]})"
    else:
        str_displacement = f"({displacement[0]}_{displacement[1]}_{displacement[2]})"

    str_pixel = f"({pixel[0]}_{pixel[1]})"
    title = f"Modal Coordinates for Displacement {str_displacement} and Pixel {str_pixel}"
    fig.suptitle(title, fontsize=16)
    fig.supylabel('Imaginary Part')
    fig.supxlabel('Real Part')

    if save:
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        filename = save_dir/f"modal_coordinates_{str_displacement}_{str_pixel}.png"
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        print(f"Modal coordinates saved to: {filename}")

    if show:
        plt.show()
    

def save_modal_coordinates(
    modal_coordinates: torch.Tensor,
    save_dir: str | PosixPath = "./",
    displacement: torch.Tensor | None = None,
    pixel: tuple[int, int] | None = None
) -> torch.Tensor:
    """
    Save the modal coordinates into a numpy file
    """
    if displacement is None:
        displacement = (0, 0)
    if pixel is None:
        pixel = (0, 0)

    modal_coordinates = modal_coordinates.detach().cpu().numpy()
    
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    
    save_dir = save_dir / "modal_coordinates"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    if displacement.shape[0] == 2:
        filename = save_dir/f"disp_{round(displacement[0].real.item(), 2)}_{round(displacement[1].real.item(), 2)}_px_{pixel[0]}_{pixel[1]}.npy"
    else:
        filename = save_dir/f"disp_{round(displacement[0].real.item(), 2)}_{round(displacement[1].real.item(), 2)}_{round(displacement[2].real.item(), 2)}_px_{pixel[0]}_{pixel[1]}.npy"
    
    np.save(filename, modal_coordinates)
    print(f"Modal coordinates saved to: {filename}")


def load_modal_coordinates(
    path: str
) -> torch.Tensor:
    """Load the modal coordinates from a numpy file
    """
    return torch.tensor(np.load(path))