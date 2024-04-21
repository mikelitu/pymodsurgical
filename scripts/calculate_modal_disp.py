import pymodal_surgical
from pymodal_surgical import motion_spectrum, optical_flow, displacement
from pathlib import Path, PosixPath
from pymodal_surgical.video_reader import VideoReader, RetType
import json
from pymodal_surgical.masking import Masking
import torch
import pymodal_surgical.complex



def main(
    video_reader: VideoReader,
    init_displacement: torch.Tensor,
    target_displacement: torch.Tensor,
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
        mode_shapes = (motion_spectrum.calculate_motion_spectrum(frames[0], K, filtered=filtering, mask=mask, camera_pos="left", save_flow_video=False), motion_spectrum.calculate_motion_spectrum(frames[1], K, filtered=filtering, mask=mask, camera_pos="right"))
        motion_frequencies = optical_flow.get_motion_frequencies(len(frames[0]), K, sampling_period=1/30) 
    else:
        mode_shapes = motion_spectrum.calculate_motion_spectrum(frames, K, filtered=filtering, mask=mask)
        motion_frequencies = optical_flow.get_motion_frequencies(len(frames), K, sampling_period=1/30)
    
    complex_mode_shapes = pymodal_surgical.complex.motion_spectrum_2_complex(mode_shapes)
    
    if isinstance(complex_mode_shapes, tuple):
        target_displacement = target_displacement.to(complex_mode_shapes[0].device, dtype=complex_mode_shapes[0].dtype)
        init_displacement = init_displacement.to(complex_mode_shapes[0].device, dtype=complex_mode_shapes[0].dtype)
        for i in range(1):
            _, modal_coordinates = displacement.calculate_deformation_map_from_displacement(complex_mode_shapes[i], init_displacement, pixel, alpha=0.5)
            deformation_map, _ = displacement.calculate_deformation_map_from_displacement(complex_mode_shapes[i], target_displacement, pixel, alpha=0.5)
            force = displacement.calculate_force_from_displacement_map(complex_mode_shapes[i], deformation_map, modal_coordinates, motion_frequencies, pixel, timestep=0.03)
            print(force)
            
    else:
        target_displacement = target_displacement.to(complex_mode_shapes.device, dtype=complex_mode_shapes.dtype)
        init_displacement = init_displacement.to(complex_mode_shapes.device, dtype=complex_mode_shapes.dtype)
        _, modal_coordinates = displacement.calculate_deformation_map_from_displacement(complex_mode_shapes, init_displacement, pixel, alpha=0.5)
        deformation_map, _ = displacement.calculate_deformation_map_from_displacement(complex_mode_shapes, target_displacement, pixel, alpha=0.5)
        force = displacement.calculate_force_from_displacement_map(complex_mode_shapes, deformation_map, modal_coordinates, motion_frequencies, pixel, timestep=0.03)
        print(force)


if __name__ == "__main__":
    video_path = Path("videos/test_video.mp4")
    K = 16
    init_displacement = torch.tensor([0.0, 0.0])
    target_displacement = torch.tensor([2.0, 0.0])
    pixel = (64, 64)
    with open("videos/metadata.json", "r") as f:
        metadata = json.load(f)
    
    reader = VideoReader(video_path, video_config=metadata, return_type=RetType.NUMPY)
    # Apply all combinations of filtering and masking
    main(reader, init_displacement, target_displacement, pixel, K=K, save_dir="spectrums", filtering=True, masking=True)
    

