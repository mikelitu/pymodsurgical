from video_reader import VideoReader
import depth
import torch
from pathlib import Path
import json
import numpy as np
import unittest

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def read_video(
    video_path: str | Path,
    metadata_path: str | Path,
    return_type: str = "numpy",
) -> VideoReader:
    with open(metadata_path, "r") as f:
        video_config = json.load(f)
    
    return VideoReader(video_path, video_config, return_type)


def init_depth(
    model_type: depth.ModelType = depth.ModelType.DPT_Large
) -> tuple[torch.nn.Module, torch.nn.Module]:
    return depth.load_depth_model_and_transform(model_type=model_type)


class TestDepth(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.depth_model, cls.depth_transform = init_depth()
        cls.depth_model.to(device)
        reader = read_video(Path("videos/liver_stereo.avi"), Path("videos/metadata.json"))
        cls.frame = reader.read_frame(0)[0] if reader.video_type == "stereo" else reader.read_frame(0)
        cls.intrinsics = reader.left_calibration_mat
        cls.frames = reader.read(0, 10)[0] if reader.video_type == "stereo" else reader.read(0, 10)

    def test_calculate_depth_map(self):
        depth_frame = depth.calculate_depth_map(self.depth_model, self.depth_transform, self.frame)
        self.assertTrue(isinstance(depth_frame, np.ndarray))
        self.assertTrue(depth_frame.squeeze(0).shape == self.frame.shape[:2])


    def test_calculate_depth_map_from_video(self):
        depth_maps = depth.calculate_depth_map_from_video(self.depth_model, self.depth_transform, self.frames)
        self.assertTrue(isinstance(depth_maps, np.ndarray))
        self.assertTrue(depth_maps.shape[0] == 10)
        self.assertTrue(depth_maps[0].shape == self.frames[0].shape[:2])
    

    def test_unproject_image_to_point_cloud(self):
        depth_frame = depth.calculate_depth_map(self.depth_model, self.depth_transform, self.frame)
        unprojected_img = depth.unproject_image_to_point_cloud(depth_frame.squeeze(0), self.intrinsics, False)
        self.assertTrue(isinstance(unprojected_img, np.ndarray))
        self.assertTrue(unprojected_img.shape[1] == 3)
    

    def test_simplify_point_cloud(self):
        depth_frame = depth.calculate_depth_map(self.depth_model, self.depth_transform, self.frame)
        unprojected_img = depth.unproject_image_to_point_cloud(depth_frame.squeeze(0), self.intrinsics, False)
        simp_unprojected_img = depth.unproject_image_to_point_cloud(depth_frame.squeeze(0), self.intrinsics, True)
        self.assertTrue(isinstance(simp_unprojected_img, np.ndarray))
        self.assertLessEqual(simp_unprojected_img.shape[0], unprojected_img.shape[0])
        self.assertTrue(simp_unprojected_img.shape[0] == 1000)


if __name__ == "__main__":
    unittest.main()
