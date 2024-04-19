from motion_spectrum import calculate_motion_spectrum, save_motion_spectrum
from pathlib import Path, PosixPath
from video_reader import VideoReader, VideoType, RetType
import json
from masking import Masking
from displacement import calculate_deformation_map_from_displacement
import torch
from complex import motion_spectrum_2_complex, save_modal_coordinates
from optical_flow import plot_and_save
from torchvision.utils import flow_to_image


def main(
    video_reader: VideoReader,
    displacement: torch.Tensor,
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
        motion_spectrum = (calculate_motion_spectrum(frames[0], K, filtered=filtering, mask=mask, camera_pos="left", save_flow_video=True), calculate_motion_spectrum(frames[1], K, filtered=filtering, mask=mask, camera_pos="right"))
    else:
        motion_spectrum = calculate_motion_spectrum(frames, K, filtered=filtering, mask=mask)
    
    if save_dir is not None:
        save_motion_spectrum(motion_spectrum, save_dir, filtered=filtering, masked=masking)
    
    complex_motion_spectrum = motion_spectrum_2_complex(motion_spectrum)
    
    if isinstance(complex_motion_spectrum, tuple):
        displacement = displacement.to(complex_motion_spectrum[0].device, dtype=complex_motion_spectrum[0].dtype)
        for i in range(2):
            deformation_map, modal_coordinates = calculate_deformation_map_from_displacement(complex_motion_spectrum[i], displacement, pixel)
            deformation_map_img = flow_to_image(deformation_map.unsqueeze(0))
            save_modal_coordinates(modal_coordinates, save_dir, displacement, pixel)
            plot_and_save(deformation_map_img, "test/displacement_map_{}.png".format(i))
    else:
        displacement = displacement.to(complex_motion_spectrum.device, dtype=complex_motion_spectrum.dtype)
        deformation_map, modal_coordinates = calculate_deformation_map_from_displacement(complex_motion_spectrum, displacement, pixel)
        deformation_map_img = flow_to_image(deformation_map.unsqueeze(0))
        plot_and_save(deformation_map_img, save_dir/"deformation_map.png")


if __name__ == "__main__":
    video_path = Path("videos/test_video.mp4")
    K = 16
    displacement = torch.tensor([1.0, 0.0])
    pixel = (64, 64)
    with open("videos/metadata.json", "r") as f:
        metadata = json.load(f)
    
    video_type = VideoType(metadata[video_path.stem]["video_type"])
    reader = VideoReader(video_path, video_config=metadata, return_type=RetType.NUMPY)
    
    # Apply all combinations of filtering and masking
    main(reader, displacement, pixel, K=K, save_dir="spectrums", filtering=True, masking=True)
    

