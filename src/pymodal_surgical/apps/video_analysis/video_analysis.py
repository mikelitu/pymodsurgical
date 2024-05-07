from pymodal_surgical.modal_analysis import deformation, functions
from pathlib import Path, PosixPath
from pymodal_surgical.video_processing.reader import VideoReader, VideoType, RetType
import json
from pymodal_surgical.video_processing.masking import Masking
from pymodal_surgical.modal_analysis import depth
import torch
from pymodal_surgical.modal_analysis.math_helper import mode_shape_2_complex
from pymodal_surgical.modal_analysis.utils import save_modal_coordinates
from pymodal_surgical.modal_analysis.plot_utils import save_mode_shape, plot_and_save
from pymodal_surgical.modal_analysis.functions import calculate_mode_shapes
from torchvision.utils import flow_to_image


class VideoAnalyzer():

    def __init__(self, config: dict) -> None:
        self._load_experiment(config)

    
    def _calculate_mode_shapes(self, frames, K, filtering, mask):
        if isinstance(frames, tuple):
            return (functions.calculate_mode_shapes(frames[0], K, filter_config=filtering, mask=mask, camera_pos="left", save_flow_video=True), functions.calculate_motion_spectrum(frames[1], K, filtered=filtering, mask=mask, camera_pos="right"))
        else:
            return functions.calculate_mode_shapes(frames, K, filtered=filtering, mask=mask)

    def _save_mode_shape(self, mode_shapes, save_dir, filtering, masked):
        if isinstance(mode_shapes, tuple):
            for i in range(2):
                functions.save_mode_shape()
        else:
            functions.save_mode_shape(mode_shapes, save_dir, filtered=filtering, masked=masked)     
    def _load_experiment(self, config):
        video_reader = VideoReader(video_config=config, return_type=RetType.NUMPY)
        self.frames = video_reader.read(int(config["start"]), int(config["end"]))
        self.depth_model, self.depth_transform = depth.load_depth_model_and_transform()
        if config["masking"]["enabled"]:
            self.mask = Masking(config["masking"]["mask"], video_reader.video_type)
        else:
            self.mask = None
        




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
        mode_shapes = (calculate_mode_shapes(frames[0], K, filtered=filtering, mask=mask, camera_pos="left", save_flow_video=True), functions.calculate_motion_spectrum(frames[1], K, filtered=filtering, mask=mask, camera_pos="right"))
    else:
        mode_shapes = calculate_mode_shapes(frames, K, filtered=filtering, mask=mask)
    
    if save_dir is not None:
        save_mode_shape(mode_shapes, save_dir, filtered=filtering, masked=masking)
    
    complex_mode_shapes = mode_shape_2_complex(mode_shapes)
    
    if isinstance(complex_mode_shapes, tuple):
        disp = disp.to(complex_mode_shapes[0].device, dtype=complex_mode_shapes[0].dtype)
        for i in range(2):
            deformation_map, modal_coordinates = deformation.calculate_deformation_map_from_displacement(complex_mode_shapes[i], deformation, pixel)
            deformation_map_img = flow_to_image(deformation_map.unsqueeze(0))
            save_modal_coordinates(modal_coordinates, save_dir, disp, pixel)
            plot_and_save(deformation_map_img, "test/displacement_map_{}.png".format(i))
    else:
        disp = disp.to(complex_mode_shapes.device, dtype=complex_mode_shapes.dtype)
        deformation_map, modal_coordinates = deformation.calculate_deformation_map_from_displacement(complex_mode_shapes, disp, pixel)
        deformation_map_img = flow_to_image(deformation_map.unsqueeze(0))
        plot_and_save(deformation_map_img, save_dir/"deformation_map.png")


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
    

