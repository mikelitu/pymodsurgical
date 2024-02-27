import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import PosixPath, Path

def tensor_rotate_phase_to_real_axis(tensor: torch.Tensor) -> torch.Tensor:
    """
    Rotate the phase of a complex tensor to the real axis.
    """
    return torch.abs(tensor)

def complex_from_magnitude_phase(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """
    Convert polar coordinates to a complex tensor.
    """
    real_part = magnitude * torch.cos(phase)
    imag_part = magnitude * torch.sin(phase)
    return real_part + 1j * imag_part

def motion_spectrum_2_complex(
    motion_spectrum: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a motion spectrum to a complex tensor.
    """
    if isinstance(motion_spectrum, tuple):
        left_motion_spectrum = motion_spectrum[0]
        right_motion_spectrum = motion_spectrum[1]
        left_complex_motion_spectrum = motion_spectrum_2_complex(left_motion_spectrum)
        right_complex_motion_spectrum = motion_spectrum_2_complex(right_motion_spectrum)
        return left_complex_motion_spectrum, right_complex_motion_spectrum
    
    K, _, height, width = motion_spectrum.shape
    complex_motion_spectrum = torch.zeros(K, 2, height, width).to(motion_spectrum.device, dtype=torch.cfloat)

    complex_motion_spectrum[:, 0, :, :] = motion_spectrum[:, 0, :, :] + 1j * motion_spectrum[:, 1, :, :]
    complex_motion_spectrum[:, 1, :, :] = motion_spectrum[:, 2, :, :] + 1j * motion_spectrum[:, 3, :, :]
    return complex_motion_spectrum

def normalize_modal_coordinate(
    modal_coordinate: torch.Tensor
) -> torch.Tensor:
    """
    Normalize the modal coordinate.
    """
    norm = torch.linalg.norm(modal_coordinate, dim=1, ord=2)
    return modal_coordinate / norm.unsqueeze(-1)

def plot_modal_coordinates(
    modal_coordinates: torch.Tensor | np.ndarray,
    displacement: torch.Tensor | None = None,
    pixel: tuple[int, int] | None = None
) -> None:
    """
    Plot the modal coordinates.
    """

    # Transform the modal coordinates to numpy
    if isinstance(modal_coordinates, torch.Tensor):
        modal_coordinates = normalize_modal_coordinate(modal_coordinates)
        modal_coordinates = modal_coordinates.detach().cpu().numpy()
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 10))
    
    # Setting the aspect ratio of the plot to 'equal to ensure scaled arrows
    ax1.set_aspect('equal', 'box')
    ax2.set_aspect('equal', 'box')

    # Plot each modal coordinate
    for mc in modal_coordinates:
        ax1.quiver(0, 0, mc[0].real, mc[0].imag, angles='xy', scale_units='xy', scale=1, color='r')
        ax2.quiver(0, 0, mc[1].real, mc[1].imag, angles='xy', scale_units='xy', scale=1, color='r')
    
    # Set the limits of the plot (we know all the coordinates are normalized)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.grid(True)
    ax1.axhline(0, color='black', lw=0.5)
    ax1.axvline(0, color='black', lw=0.5)
    ax1.set_xlabel('Real Part')
    ax1.set_ylabel('Imaginary Part')
    ax1.set_title('Modal Coordinates X')

    # Set the labels of the plot
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True)
    ax2.axhline(0, color='black', lw=0.5)
    ax2.axvline(0, color='black', lw=0.5)
    ax2.set_xlabel('Real Part')
    ax2.set_ylabel('Imaginary Part')
    ax2.set_title('Modal Coordinates Y')

    fig.suptitle(f"Displacement: ({displacement[0]}, {displacement[1]}) Pixel: ({pixel[0]}, {pixel[1]})")
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
    

    np.save(save_dir/f"disp_{displacement[0].real}_{displacement[1].real}_px_{pixel[0]}_{pixel[1]}.npy", modal_coordinates)

def load_modal_coordinates(
    path: str
) -> torch.Tensor:
    """Load the modal coordinates from a numpy file
    """
    return torch.tensor(np.load(path))