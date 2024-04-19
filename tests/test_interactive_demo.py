# import unittest
# from pathlib import Path
# from interactive_demo import InteractiveDemo, ControlType
# import depth
# import torch

# class TestInteractiveDemo(unittest.TestCase):

#     def setUp(self):
#         video_path = Path("/path/to/video.mp4")
#         self.demo = InteractiveDemo(video_path)

#     def test_init(self):
#         self.assertIsInstance(self.demo, InteractiveDemo)
#         self.assertEqual(self.demo.solver_time_step, 0.03)
#         self.assertEqual(self.demo.fps, 24)
#         self.assertEqual(self.demo.control_type, ControlType.MOUSE)
#         self.assertEqual(self.demo.norm_scale, 10.0)
#         self.assertEqual(self.demo.alpha_limit, 5.0)
#         self.assertEqual(self.demo.force_scale, 0.35)
#         self.assertEqual(self.demo.display_cropping, (30, 30))
#         self.assertEqual(self.demo.near, 0.05)
#         self.assertEqual(self.demo.far, 0.95)
#         self.assertFalse(self.demo.inverse)
#         self.assertEqual(self.demo.rayleigh_mass, 0.1)
#         self.assertEqual(self.demo.rayleigh_stiffness, 0.1)

#     def test_init_pygame(self):
#         self.demo._init_pygame(self.demo.reference_frame)
#         self.assertIsNotNone(self.demo.screen)
#         self.assertIsNotNone(self.demo.image)
#         self.assertTrue(self.demo.running)
#         self.assertFalse(self.demo.on_click)
#         self.assertIsNone(self.demo.pixel)
#         self.assertEqual(self.demo.cur_pos, [0, 0])
#         self.assertEqual(self.demo.alpha, 0)
#         self.assertIsNotNone(self.demo.clock)
#         self.assertEqual(self.demo.displacement, torch.tensor([0, 0]))

#     def test_init_haptic_device(self):
#         self.demo._init_haptic_device()
#         self.assertIsNotNone(self.demo.device)
#         self.assertIsNotNone(self.demo.init_position)
#         self.assertFalse(self.demo.pre_button)

#     def test_scale_haptic_2_screen(self):
#         position = torch.tensor([100, 200])
#         scaled_position = self.demo._scale_haptic_2_screen(position)
#         self.assertEqual(scaled_position, (50, 0))  # Replace with expected values

#     def test_haptic_control(self):
#         # TODO: Write test for haptic_control method
#         pass

#     def test_mouse_control(self):
#         # TODO: Write test for mouse_control method
#         pass

#     def test_sim_step(self):
#         # TODO: Write test for sim_step method
#         pass

# if __name__ == '__main__':
#     unittest.main()