import torch
import optical_flow
import unittest
import numpy as np

device = torch.device("cpu")

class TestOpticalFlow(unittest.TestCase):
    def test_motion_texture_from_flow_field(self):
        """We are checking if the function returns the correct shape.
        The only factor that it should be fullfilled is that the timstep should be >> K."""
        # FILEPATH: /home/md21/real-surgery-videos/tests/test_optical_flow.py
        # Test case 1: 2D flow field
        timestep = 25
        K = 6
        flow_field = torch.randn(2, timestep, 2, 4, 4)
        motion_texture = optical_flow.motion_texture_from_flow_field(flow_field, K, timestep)
        self.assertEqual(motion_texture.shape, (2, K, 4, 4, 4))

        # Test case 2: £D flow field
        flow_field = torch.randn(1, timestep, 3, 5, 5)
        K = 4
        motion_texture = optical_flow.motion_texture_from_flow_field(flow_field, K, timestep)
        self.assertEqual(motion_texture.shape, (1, K, 6, 5, 5))

        # Test case 3: ValueError
        flow_field = torch.randn(3, timestep, 8, 3, 3)
        K = 4
        with self.assertRaises(ValueError):
            motion_texture = optical_flow.motion_texture_from_flow_field(flow_field, K, timestep)


    def test_flow_field_from_motion_texture(self):
        # Test case 1: 2D motion texture
        timesteps = 25
        K = 6
        motion_texture = torch.randn(2, K, 4, 4, 4)
        flow_field = optical_flow.flow_field_from_motion_texture(motion_texture, timesteps, K)
        self.assertEqual(flow_field.shape, (2, timesteps, 2, 4, 4))

        # Test case 2: 3D motion texture
        motion_texture = torch.randn(1, K, 6, 5, 5)
        flow_field = optical_flow.flow_field_from_motion_texture(motion_texture, timesteps, K)
        self.assertEqual(flow_field.shape, (1, timesteps, 3, 5, 5))

        # Test case 3: ValueError
        motion_texture = torch.randn(3, K, 8, 3, 3)
        with self.assertRaises(ValueError):
            flow_field = optical_flow.flow_field_from_motion_texture(motion_texture, timesteps, K)


    def test_estimate_flow(self):
        # Test case 1: 2D flow field
        model = optical_flow.load_flow_model(device)
        reference_frames = torch.randn(2, 3, 4, 4)
        target_sequences = torch.randn(2, 5, 3, 4, 4)
        flows = optical_flow.estimate_flow(model, reference_frames, target_sequences)
        self.assertEqual(flows.shape, (2, 5, 2, 4, 4))

        # Test case 2: 3D flow field
        reference_frames = torch.randn(1, 3, 5, 5)
        target_sequences = torch.randn(1, 6, 3, 5, 5)
        flows = optical_flow.estimate_flow(model, reference_frames, target_sequences)
        self.assertEqual(flows.shape, (1, 6, 2, 5, 5))

        # Test case 3: Upscaling
        reference_frames = torch.randn(1, 3, 100, 100)
        target_sequences = torch.randn(1, 4, 3, 100, 100)
        flows = optical_flow.estimate_flow(model, reference_frames, target_sequences)
        self.assertEqual(flows.shape, (1, 4, 2, 100, 100))
    

    def test_preprocess_for_raft(self):
        # Test case 1: Single batch
        batch = np.random.randn(256, 256, 3)
        expected_output_shape = (3, 128, 128)
        preprocessed_batch = optical_flow.preprocess_for_raft(batch)
        self.assertEqual(preprocessed_batch.shape, expected_output_shape)
        self.assertTrue(isinstance(preprocessed_batch, torch.Tensor))

        # Test case 2: Multiple batches
        batch = np.random.randn(2, 256, 256, 3)
        expected_output_shape = (2, 3, 128, 128)
        preprocessed_batch = torch.stack([optical_flow.preprocess_for_raft(b) for b in batch])
        self.assertEqual(preprocessed_batch.shape, expected_output_shape)
        self.assertTrue(isinstance(preprocessed_batch, torch.Tensor))

        # Test case 3: Empty batch returns TypeError
        batch = torch.empty(0, 3, 256, 256)
        expected_output_shape = (0, 3, 128, 128)
        with self.assertRaises(TypeError):
            preprocessed_batch = optical_flow.preprocess_for_raft(batch)
    

    def test_make_grid(self):
        # Test case 1: Single image
        img = torch.randn(1, 3, 256, 256)
        grid = optical_flow.make_grid(img)
        self.assertEqual(grid.shape, (1, 2, 256, 256))
        self.assertTrue(torch.allclose(grid[:, 0], torch.arange(0, 256).view(1, 1, 256).float()))
        self.assertTrue(torch.allclose(grid[:, 1], torch.arange(0, 256).view(1, 256, 1).float()))

        # Test case 2: Multiple images
        img = torch.randn(2, 3, 128, 128)
        grid = optical_flow.make_grid(img)
        self.assertEqual(grid.shape, (2, 2, 128, 128))
        self.assertTrue(torch.allclose(grid[:, 0], torch.arange(0, 128).view(1, 1, 128).float()))
        self.assertTrue(torch.allclose(grid[:, 1], torch.arange(0, 128).view(1, 128, 1).float()))

        # Test case 3: Empty image returns empty grid
        img = torch.empty(0, 3, 256, 256)
        grid = optical_flow.make_grid(img)
        self.assertEqual(grid.shape, (0, 2, 256, 256))

        # Test case 4: Bad input type returns AttributeError from .to()
        img = np.random.randn(1, 3, 256, 256)
        with self.assertRaises(AttributeError):
            grid = optical_flow.make_grid(img)
    

    # def test_warp_flow(self):
    #     # Test case 1: Single channel image, 2D flow
    #     img = 255 * torch.tensor([[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]])
    #     flow = torch.tensor([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]])
    #     print(img.shape, flow.shape)
    #     depth_map = np.ones((3, 3))
    #     warped_img = optical_flow.warp_flow(img, flow, depth_map, far=1.0, near=0.0)
    #     expected_warped_img = 255 * torch.tensor([[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]).permute(1, 2, 0).cpu().numpy() 
    #     self.assertTrue(torch.allclose(warped_img, expected_warped_img, atol=1e-2))

    #     # Test case 2: RGB image, 3D flow
    #     img = torch.tensor([[[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]])
    #     flow = torch.tensor([[[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]]])
    #     depth_map = np.ones((1, 3, 3))
    #     warped_img = optical_flow.warp_flow(img, flow, depth_map)
    #     expected_warped_img = np.array([[[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]])
    #     self.assertTrue(np.testing.assert_array_almost_equal(warped_img, expected_warped_img))

    #     # Test case 3: Inverse warp
    #     img = torch.tensor([[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]])
    #     flow = torch.tensor([[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]])
    #     depth_map = np.ones((3, 3))
    #     warped_img = optical_flow.warp_flow(img, flow, depth_map, inverse=True)
    #     expected_warped_img = np.array([[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]])
    #     self.assertTrue(np.testing.assert_array_almost_equal(warped_img, expected_warped_img))

    #     # Test case 4: Masked warp
    #     img = torch.tensor([[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]])
    #     flow = torch.tensor([[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]])
    #     depth_map = np.ones((3, 3))
    #     mask = np.ones((3, 3))
    #     warped_img = optical_flow.warp_flow(img, flow, depth_map, mask=mask)
    #     expected_warped_img = np.array([[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]])
    #     self.assertTrue(np.testing.assert_array_almost_equal(warped_img, expected_warped_img))


if __name__ == "__main__":
    unittest.main()