import torch
import numpy as np
from pathlib import PosixPath, Path


def save_modal_coordinates(
    modal_coordinates: torch.Tensor,
    save_dir: str | PosixPath = "./",
    displacement: torch.Tensor | None = None,
    pixel: tuple[int, int] | None = None
) -> torch.Tensor:
    """
    Save the modal coordinates into a numpy file.

    Args:
        modal_coordinates (torch.Tensor): The modal coordinates to be saved.
        save_dir (str | PosixPath, optional): The directory to save the file. Defaults to "./".
        displacement (torch.Tensor | None, optional): The displacement values. Defaults to None.
        pixel (tuple[int, int] | None, optional): The pixel values. Defaults to None.

    Returns:
        torch.Tensor: The saved modal coordinates.

    Raises:
        None

    Examples:
        >>> modal_coordinates = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> save_modal_coordinates(modal_coordinates, save_dir="./data", displacement=torch.tensor([0, 0]), pixel=(100, 100))
        Modal coordinates saved to: ./data/modal_coordinates/disp_0.0_0.0_px_100_100.npy
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
    """Load the modal coordinates from a numpy file.

    Args:
        path (str): The path to the numpy file.

    Returns:
        torch.Tensor: The modal coordinates loaded as a PyTorch tensor.
    """
    return torch.tensor(np.load(path))