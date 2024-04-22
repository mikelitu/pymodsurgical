import numpy as np
import torch
from pymodal_surgical.video_processing.filtering import GaussianFiltering
import unittest

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class TestFiltering(unittest.TestCase):
    def test_compute_local_contrast(self):
        img = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        gf = GaussianFiltering()
        result = gf._compute_local_contrast(img)
        self.assertEqual(result.shape, (100, 100))
        self.assertLessEqual(result.max(), 1.0)
        self.assertGreaterEqual(result.min(), 0.0)

    def test_blur_image(self):
        img = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        gf = GaussianFiltering()
        result = gf._blur_image(img)
        self.assertEqual(result.shape, (100, 100, 3))

    def test_filter_image(self):
        img = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        flow = np.random.randn(100, 100, 2)
        gf = GaussianFiltering()
        result = gf._filter_image(img, flow)
        self.assertEqual(result.shape, (100, 100, 2))
        self.assertLessEqual(result.max(), 1.0)
        self.assertGreaterEqual(result.min(), 0.0)

    def test_filter_video(self):
        video = [np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8) for _ in range(10)]
        flow = [np.random.randn(100, 100, 2) for _ in range(10)]
        gf = GaussianFiltering()
        result = gf._filter_video(video, flow)
        expected_shape = (100, 100, 2)
        self.assertEqual(len(result), 10)
        self.assertCountEqual([frame.shape for frame in result], [expected_shape] * len(result))
        for frame in result:
            self.assertLessEqual(frame.max(), 1.0)
            self.assertGreaterEqual(frame.min(), 0.0)
        
    def test_filter_stereo_video(self):
        video = ([np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8) for _ in range(10)],
                [np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8) for _ in range(10)])
        flow = ([np.random.randn(100, 100, 2) for _ in range(10)],
                [np.random.randn(100, 100, 2) for _ in range(10)])
        gf = GaussianFiltering()
        result = gf._filter_stereo_video(video, flow)
        self.assertEqual(len(result), 2)
        self.assertCountEqual([len(frame) for frame in result], [10] * 2)
        self.assertCountEqual([frame.shape for frame in result[0]], [(100, 100, 2)] * 10)
        self.assertCountEqual([frame.shape for frame in result[1]], [(100, 100, 2)] * 10)
        for frame in result[0]:
            self.assertLessEqual(frame.max(), 1.0)
            self.assertGreaterEqual(frame.min(), 0.0)
        for frame in result[1]:
            self.assertLessEqual(frame.max(), 1.0)
            self.assertGreaterEqual(frame.min(), 0.0)

    def test_filter_mono_video(self):
        video = [np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8) for _ in range(10)]
        flow = [np.random.randn(100, 100, 2) for _ in range(10)]
        gf = GaussianFiltering()
        result = gf._filter_mono_video(video, flow)
        self.assertEqual(len(result), 10)
        self.assertCountEqual([frame.shape for frame in result], [(100, 100, 2)] * 10)
        for frame in result:
            self.assertLessEqual(frame.max(), 1.0)
            self.assertGreaterEqual(frame.min(), 0.0)

    def test_call(self):
        img = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        flow = np.random.randn(100, 100, 2)
        gf = GaussianFiltering()
        result = gf(img, flow)
        self.assertEqual(result.shape, (1, 100, 100, 2))
        self.assertLessEqual(result.max(), 1.0)
        self.assertGreaterEqual(result.min(), 0.0)

        img = (torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0).to(device)
        flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float().to(device)
        result = gf(img, flow)
        self.assertEqual(result.shape, (1, 2, 100, 100))
        self.assertLessEqual(result.max(), 1.0)
        self.assertGreaterEqual(result.min(), 0.0)

        
    def test_errors(self):
        video = ([np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8) for _ in range(10)],
                [np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8) for _ in range(10)])
        flow = [np.random.randn(100, 100, 2) for _ in range(10)]
        gf = GaussianFiltering()
        with self.assertRaises(ValueError):
            gf(video, flow)
        
        video = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        flow = (np.random.randn(100, 100, 2), np.random.randn(100, 100, 2))
        with self.assertRaises(ValueError):
            gf(video, flow)
        
        video = (torch.from_numpy(np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0).float() / 255.0).to(device)
        flow = (torch.from_numpy(np.random.randn(100, 100, 2)).permute(2, 0, 1).unsqueeze(0).float().to(device),
                torch.from_numpy(np.random.randn(100, 100, 2)).permute(2, 0, 1).unsqueeze(0).float().to(device))
        with self.assertRaises(ValueError):
            gf(video, flow)
        
        video = (torch.from_numpy(np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0).float() / 255.0).to(device)
        flow = np.random.randn(100, 100, 2)
        with self.assertRaises(ValueError):
            gf(video, flow)
        
        video = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        flow = torch.randn(100, 100, 2)
        with self.assertRaises(ValueError):
            gf(video, flow)


if __name__ == "__main__":
    unittest.main()