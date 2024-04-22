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