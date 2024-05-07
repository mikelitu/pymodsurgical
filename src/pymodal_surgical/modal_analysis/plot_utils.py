import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from pathlib import PosixPath, Path
from . import math_helper
from .. import utils


def plot_and_save(imgs: list[torch.Tensor] | torch.Tensor, filename: PosixPath | None = None, **imshow_kwargs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    
    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 5, num_rows * 5), squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx][col_idx]
            img = torchvision.transforms.functional.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def mode_shape_2_grayimage(motion_spectrum: torch.Tensor) -> torch.Tensor:
    """
    Converts a motion spectrum to a RGB image.

    Args:
        motion_texture (torch.Tensor): Spectrum of shape (K, 4, H, W) and dtype torch.float.

    Returns:
        img (Tensor): Image Tensor of dtype uint8 and shape where each color corresponds to 
        a given spectrum direction. Shape is (K, 3, H, W).
    """

    if motion_spectrum.dtype != torch.float:
        raise ValueError("The motion spectrum should have dtype torch.float")
    
    orig_shape = motion_spectrum.shape
    if motion_spectrum.ndim != 4 or (orig_shape[1] != 4 and orig_shape[1] != 6):
        raise ValueError("The motion spectrum should have shape (K, 4, H, W) or (K, 6, H, W)")
    
    motion_spectrum_X = torch.abs(motion_spectrum[:, 0] + 1j * motion_spectrum[:, 1])
    motion_spectrum_Y = torch.abs(motion_spectrum[:, 2] + 1j * motion_spectrum[:, 3])
    if orig_shape[1] == 6:
        motion_spectrum_Z = torch.abs(motion_spectrum[:, 4] + 1j * motion_spectrum[:, 5])
        motion_spectrum = torch.stack([motion_spectrum_X, motion_spectrum_Y, motion_spectrum_Z], dim=1)
    else:
        motion_spectrum = torch.stack([motion_spectrum_X, motion_spectrum_Y], dim=1)
    max_norm = torch.sum(motion_spectrum**2, dim=1).sqrt().max()
    epsilon = torch.finfo(motion_spectrum.dtype).eps
    motion_spectrum = motion_spectrum / (max_norm + epsilon)
    # img = _normalized_motion_spectrum_to_image(motion_spectrum)
    if orig_shape[1] == 6:
        return motion_spectrum[:, 0], motion_spectrum[:, 1], motion_spectrum[:, 2]
    
    return motion_spectrum[:, 0], motion_spectrum[:, 1]


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
        modal_coordinates = math_helper.normalize_modal_coordinate(modal_coordinates)
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


def save_mode_shape(
    motion_spectrum: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    save_dir: PosixPath | str,
    filtered: bool = False,
    masked: bool = False
) -> None:
    
    if isinstance(motion_spectrum, tuple):
        filenames = {0: "left_motion_spectrum.png", 1: "right_motion_spectrum.png"}
        for i in range(len(motion_spectrum)):
            filename = filenames[i]
            if filtered:
                filename = "filtered_" + filename
            if masked:
                filename = "masked_" + filename
                
            tmp_save_dir = utils.create_save_dir(save_dir, filename)
            print(f"Saving motion spectrum to: {tmp_save_dir}")
            dim = motion_spectrum[i].shape[1]
            if dim == 4:
                img_motion_spectrum_X, img_motion_spectrum_Y = mode_shape_2_grayimage(motion_spectrum[i])
                plot_and_save([img_motion_spectrum_X, img_motion_spectrum_Y], tmp_save_dir, cmap="plasma")
            else:
                img_motion_spectrum_X, img_motion_spectrum_Y, img_motion_spectrum_Z = mode_shape_2_grayimage(motion_spectrum[i])
                plot_and_save([img_motion_spectrum_X, img_motion_spectrum_Y, img_motion_spectrum_Z], tmp_save_dir, cmap="plasma")
    else:
        filename = "motion_spectrum.png"
        if filtered:
            filename = "filtered_" + filename
        if masked:
            filename = "masked_" + filename
        save_dir = utils.create_save_dir(save_dir, filename)
        print("Saving motion spectrum to: ", save_dir)
        dim = motion_spectrum.shape[1]
        if dim == 4:
            img_motion_spectrum_X, img_motion_spectrum_Y = mode_shape_2_grayimage(motion_spectrum)
            plot_and_save([img_motion_spectrum_X, img_motion_spectrum_Y], save_dir, cmap="plasma")
        else:
            img_motion_spectrum_X, img_motion_spectrum_Y, img_motion_spectrum_Z = mode_shape_2_grayimage(motion_spectrum)
            plot_and_save([img_motion_spectrum_X, img_motion_spectrum_Y, img_motion_spectrum_Z], save_dir, cmap="plasma")


def save_complex_mode_shape(
    mode_shapes: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    save_dir: PosixPath | str
) -> None:

    if isinstance(mode_shapes, tuple):
        filenames = {0: "left_complex_motion_spectrum.png", 1: "right_complex_motion_spectrum.png"}
        for i in range(mode_shapes[0].shape[0]):
            filename = filenames[i]
            tmp_save_dir = utils.create_save_dir(save_dir, filename)
            print(f"Saving complex motion spectrum to: {tmp_save_dir}")
            plot_and_save(mode_shapes[i], tmp_save_dir, cmap="plasma")
    else:
        for i in range(mode_shapes.shape[0]):
            mode_shape = mode_shapes[i]
            filename = f"mode_{i}.png"
            tmp_save_dir = utils.create_save_dir(save_dir, filename)
            np_mode_shape = mode_shape.detach().cpu().permute(1, 2, 0).numpy()
            img_mode_shape = Image.fromarray(np_mode_shape, "RGBA")
            img_mode_shape.save(tmp_save_dir)

