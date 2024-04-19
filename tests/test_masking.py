import unittest
import numpy as np
from PIL import Image
from masking import Masking
import utils
import shutil

def create_and_save_mask():
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[128:192, 128:192] = 255
    mask_save_dir = "./tests/figures"
    mask_name = "dummy_mask.png"
    mask_path = utils.create_save_dir(mask_save_dir, mask_name)

    Image.fromarray(mask).save(mask_path)
    return mask_path

class TestMasking(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.mask_path = create_and_save_mask()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.mask_path.parent)
        
    def test_apply_mask(self):
        # Create a Masking instance
        mask_path = self.mask_path
        masking = Masking(mask_path)

        # Create some sample frames
        frames = np.random.rand(3, 3, 256, 256)  # shape: (batch_size, channels, height, width)

        # Apply the mask to the frames
        masked_frames = masking.apply_mask(frames)

        # Assert that the shape of the masked frames is the same as the input frames
        self.assertEqual(masked_frames.shape, frames.shape)

        # Assert that the masked frames are multiplied by the mask
        expected_frames = frames * masking.mask[None, None, ...]
        self.assertSequenceEqual(masked_frames.tolist(), expected_frames.tolist())

    def test_call(self):
        # Create a Masking instance
        mask_path = self.mask_path
        masking = Masking(mask_path)

        # Create some sample frames
        frames = np.random.rand(3, 3, 256, 256)  # shape: (batch_size, channels, height, width)

        # Call the Masking instance
        masked_frames = masking(frames)

        # Assert that the shape of the masked frames is the same as the input frames
        self.assertEqual(masked_frames.shape, frames.shape)

        # Assert that the masked frames are multiplied by the mask
        expected_frames = frames * masking.mask[None, None, ...]
        self.assertSequenceEqual(masked_frames.tolist(), expected_frames.tolist())

if __name__ == '__main__':
    unittest.main()