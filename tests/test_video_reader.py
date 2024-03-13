from video_reader import VideoReader, RetType
import json
from pathlib import Path
import unittest
import torch
import numpy as np


class TestVideoReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        stereo_video_path = "videos/liver_stereo.avi"
        mono_video_path = "videos/capture1.avi"
        with open("videos/metadata.json", "r") as f:
            video_config = json.load(f)

        cls.stereo_video_reader_np = VideoReader(stereo_video_path, video_config, return_type=RetType.NUMPY)
        cls.stereo_video_reader_torch = VideoReader(stereo_video_path, video_config, return_type=RetType.TENSOR)
        cls.mono_video_reader_np = VideoReader(mono_video_path, video_config, return_type=RetType.NUMPY)
        cls.mono_video_reader_torch = VideoReader(mono_video_path, video_config, return_type=RetType.TENSOR)


    def test_read_frame_stereo(self):
        frame_np = self.stereo_video_reader_np.read_frame(0)
        frame_torch = self.stereo_video_reader_torch.read_frame(0)
        self.assertTrue(isinstance(frame_np, tuple))
        self.assertTrue(isinstance(frame_torch, tuple))
        self.assertTrue(isinstance(frame_np[0], np.ndarray))
        self.assertTrue(isinstance(frame_torch[0], torch.Tensor))
        self.assertFalse(isinstance(type(frame_np[0]), type(frame_torch[0])))
        self.assertTrue(frame_np[0].shape[0] == frame_torch[0].shape[1] and frame_np[0].shape[1] == frame_torch[0].shape[2])
        self.assertTrue(frame_np[1].shape[0] == frame_torch[1].shape[1] and frame_np[1].shape[1] == frame_torch[1].shape[2])
    

    def test_read_frame_mono(self):
        frame_np = self.mono_video_reader_np.read_frame(0)
        frame_torch = self.mono_video_reader_torch.read_frame(0)
        self.assertTrue(isinstance(frame_np, np.ndarray))
        self.assertTrue(isinstance(frame_torch, torch.Tensor))
        self.assertFalse(isinstance(type(frame_np), type(frame_torch)))
        self.assertTrue(frame_np.shape[0] == frame_torch.shape[1] and frame_np.shape[1] == frame_torch.shape[2])
    
    def test_read_stereo(self):
        frames_np = self.stereo_video_reader_np.read(0, 10)
        frames_torch = self.stereo_video_reader_torch.read(0, 10)
        self.assertTrue(len(frames_np[0]) == 10)
        self.assertTrue(len(frames_torch[0]) == 10)
        self.assertTrue(isinstance(frames_np[0], np.ndarray))
        self.assertTrue(isinstance(frames_torch[0], torch.Tensor))
        self.assertFalse(isinstance(type(frames_np[0]), type(frames_torch[0])))
        self.assertTrue(frames_np[0].shape[1] == frames_torch[0].shape[2] and frames_np[0].shape[2] == frames_torch[0].shape[3])

    def test_read_mono(self):
        frames_np = self.mono_video_reader_np.read(0, 10)
        frames_torch = self.mono_video_reader_torch.read(0, 10)
        self.assertTrue(len(frames_np) == 10)
        self.assertTrue(len(frames_torch) == 10)
        self.assertTrue(isinstance(frames_np, np.ndarray))
        self.assertTrue(isinstance(frames_torch, torch.Tensor))
        self.assertFalse(isinstance(type(frames_np), type(frames_torch)))
        self.assertTrue(frames_np[0].shape[0] == frames_torch[0].shape[1] and frames_np[0].shape[1] == frames_torch[0].shape[2])