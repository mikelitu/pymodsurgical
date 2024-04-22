import torch
from pymodal_surgical.modal_analysis import optical_flow
import unittest
import numpy as np

device = torch.device("cpu")

class TestOpticalFlow(unittest.TestCase):
    

    def test_estimate_flow(self):
        # Test case 1: 2D flow field
        model = optical_flow.load_flow_model(device)
        reference_frames = torch.randn(2, 3, 4, 4)
        target_sequences = torch.randn(2, 3, 4, 4)
        flows = optical_flow.estimate_flow(model, reference_frames, target_sequences)
        self.assertEqual(flows.shape, (2, 2, 4, 4))

        # Test case 2: 3D flow field
        reference_frames = torch.randn(6, 3, 5, 5)
        target_sequences = torch.randn(6, 3, 5, 5)
        flows = optical_flow.estimate_flow(model, reference_frames, target_sequences)
        self.assertEqual(flows.shape, (6, 2, 5, 5))

        # Test case 3: Upscaling
        reference_frames = torch.randn(4, 3, 100, 100)
        target_sequences = torch.randn(4, 3, 100, 100)
        flows = optical_flow.estimate_flow(model, reference_frames, target_sequences)
        self.assertEqual(flows.shape, (4, 2, 100, 100))
    

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


if __name__ == "__main__":
    unittest.main()