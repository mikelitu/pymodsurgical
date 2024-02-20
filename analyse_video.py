from motion_spectrum import calculate_motion_spectrum, save_motion_spectrum
from pathlib import Path, PosixPath
from video_reader import VideoReader, VideoType, RetType
import json


def main(
    video_reader: VideoReader,
    start: int = 0, 
    end: int = 0, 
    K: int = 16, 
    save_dir: PosixPath | str = "./",
    filtering: bool = True,
    masking: bool = False
):
    
    save_dir = Path(save_dir)/video_reader.video_path.stem
    frames = video_reader.read(start, end)


    if isinstance(frames, tuple):
        motion_spectrum = (calculate_motion_spectrum(frames[0], K, filtering=filtering, masking=masking), calculate_motion_spectrum(frames[1], K, filtering=filtering, masking=masking))
    else:
        motion_spectrum = calculate_motion_spectrum(frames, K, filtering=filtering, masking=masking)
    
    if save_dir is not None:
        save_motion_spectrum(motion_spectrum, save_dir)


if __name__ == "__main__":
    video_path = Path("videos/liver_stereo.avi")
    with open("videos/metadata.json", "r") as f:
        metadata = json.load(f)
    
    video_type = VideoType(metadata[video_path.stem]["video_type"])
    reader = VideoReader(video_path, video_type=video_type, return_type=RetType.NUMPY)
    main(reader, K=16, save_dir="spectrums")
    

