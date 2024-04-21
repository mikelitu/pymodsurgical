from pymodal_surgical import motion_spectrum, displacement, optical_flow
from pathlib import Path, PosixPath
from pymodal_surgical.video_reader import VideoReader, VideoType, RetType
import json
from pymodal_surgical.masking import Masking
import torch
from pymodal_surgical.complex import motion_spectrum_2_complex, save_modal_coordinates
from torchvision.utils import flow_to_image


def main(
    video_reader: VideoReader,
    disp: torch.Tensor,
    pixel: tuple[int, int],
    start: int = 0, 
    end: int = 0, 
    K: int = 16,
    save_dir: PosixPath | str = "./",
    filtering: bool = True,
    masking: bool = False
) -> None:
    
    save_dir = Path(save_dir)/video_reader.video_path.stem
    frames = video_reader.read(start, end)

    if masking:
        mask = Masking(video_reader.video_config[video_reader.video_path.stem]["mask"], video_reader.video_type)
    else:
        mask = None

    if isinstance(frames, tuple):
        mode_shapes = (motion_spectrum.calculate_motion_spectrum(frames[0], K, filtered=filtering, mask=mask, camera_pos="left", save_flow_video=True), motion_spectrum.calculate_motion_spectrum(frames[1], K, filtered=filtering, mask=mask, camera_pos="right"))
    else:
        mode_shapes = motion_spectrum.calculate_motion_spectrum(frames, K, filtered=filtering, mask=mask)
    
    if save_dir is not None:
        motion_spectrum.save_motion_spectrum(mode_shapes, save_dir, filtered=filtering, masked=masking)
    
    complex_mode_shapes = motion_spectrum_2_complex(mode_shapes)
    
    if isinstance(complex_mode_shapes, tuple):
        disp = disp.to(complex_mode_shapes[0].device, dtype=complex_mode_shapes[0].dtype)
        for i in range(2):
            deformation_map, modal_coordinates = displacement.calculate_deformation_map_from_displacement(complex_mode_shapes[i], displacement, pixel)
            deformation_map_img = flow_to_image(deformation_map.unsqueeze(0))
            save_modal_coordinates(modal_coordinates, save_dir, disp, pixel)
            optical_flow.plot_and_save(deformation_map_img, "test/displacement_map_{}.png".format(i))
    else:
        disp = disp.to(complex_mode_shapes.device, dtype=complex_mode_shapes.dtype)
        deformation_map, modal_coordinates = displacement.calculate_deformation_map_from_displacement(complex_mode_shapes, disp, pixel)
        deformation_map_img = flow_to_image(deformation_map.unsqueeze(0))
        optical_flow.plot_and_save(deformation_map_img, save_dir/"deformation_map.png")


if __name__ == "__main__":
    video_path = Path("videos/test_video.mp4")
    K = 16
    disp = torch.tensor([1.0, 0.0])
    pixel = (64, 64)
    with open("videos/metadata.json", "r") as f:
        metadata = json.load(f)
    
    video_type = VideoType(metadata[video_path.stem]["video_type"])
    reader = VideoReader(video_path, video_config=metadata, return_type=RetType.NUMPY)
    
    # Apply all combinations of filtering and masking
    main(reader, disp, pixel, K=K, save_dir="spectrums", filtering=True, masking=True)
    

