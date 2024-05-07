import unittest
import torch
from pymodal_surgical.modal_analysis import motion

class TestMotion(unittest.TestCase):
    def test_motion_texture_from_flow_field(self):
        """We are checking if the function returns the correct shape.
        The only factor that it should be fullfilled is that the timstep should be >> K."""
        # FILEPATH: /home/md21/real-surgery-videos/tests/test_optical_flow.py
        # Test case 1: 2D flow field
        timestep = 25
        K = 6
        flow_field = torch.randn(2, timestep, 2, 4, 4)
        mode_shapes = motion.mode_shapes_from_optical_flow(flow_field, K, timestep)
        self.assertEqual(mode_shapes.shape, (2, K, 4, 4, 4))

        # Test case 2: £D flow field
        flow_field = torch.randn(1, timestep, 3, 5, 5)
        K = 4
        mode_shapes = motion.mode_shapes_from_optical_flow(flow_field, K, timestep)
        self.assertEqual(mode_shapes.shape, (1, K, 6, 5, 5))

        # Test case 3: ValueError
        flow_field = torch.randn(3, timestep, 8, 3, 3)
        K = 4
        with self.assertRaises(ValueError):
            mode_shapes = motion.mode_shapes_from_optical_flow(flow_field, K, timestep)
