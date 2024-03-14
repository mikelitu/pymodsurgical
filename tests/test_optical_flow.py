import torch
from optical_flow import motion_texture_from_flow_field
import unittest

class TestOpticalFlow(unittest.TestCase):
    def test_motion_texture_from_flow_field(self):
        """We are checking if the function returns the correct shape.
        The only factor that it should be fullfilled is that the timstep should be >> K."""
        # FILEPATH: /home/md21/real-surgery-videos/tests/test_optical_flow.py
        # Test case 1: 2D flow field
        timestep = 25
        K = 6
        flow_field = torch.randn(2, timestep, 2, 4, 4)
        motion_texture = motion_texture_from_flow_field(flow_field, K, timestep)
        self.assertEqual(motion_texture.shape, (2, K, 4, 4, 4))

        # Test case 2: £D flow field
        flow_field = torch.randn(1, timestep, 3, 5, 5)
        K = 4
        motion_texture = motion_texture_from_flow_field(flow_field, K, timestep)
        self.assertEqual(motion_texture.shape, (1, K, 6, 5, 5))

        # Test case 3: ValueError
        flow_field = torch.randn(3, timestep, 8, 3, 3)
        K = 4
        with self.assertRaises(ValueError):
            motion_texture = motion_texture_from_flow_field(flow_field, K, timestep)


if __name__ == "__main__":
    unittest.main()