import unittest
import torch
import numpy as np
import pymodal_surgical.modal_analysis.math_helper as math_helper

class TestComplex(unittest.TestCase):
    def test_tensor_rotate_phase_to_real_axis(self):
        tensor = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j])
        real_axis = math_helper.tensor_rotate_phase_to_real_axis(tensor)
        self.assertTrue(torch.allclose(real_axis, torch.tensor([1.4142, 1.4142, 1.4142, 1.4142])))

    def test_complex_from_magnitude_phase(self):
        magnitude = torch.tensor([1.4142, 1.4142, 1.4142, 1.4142])
        phase = torch.tensor([np.pi/4, -np.pi/4, 3*np.pi/4, -3*np.pi/4])
        complex_tensor = math_helper.complex_from_magnitude_phase(magnitude, phase)
        self.assertTrue(torch.allclose(complex_tensor, torch.tensor([1+1j, 1-1j, -1+1j, -1-1j])))

    def test_complex_to_magnitude_phase(self):
        complex_tensor = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j])
        magnitude, phase = math_helper.complex_to_magnitude_phase(complex_tensor)
        self.assertTrue(torch.allclose(magnitude, torch.tensor([1.4142, 1.4142, 1.4142, 1.4142])))
        self.assertTrue(torch.allclose(phase, torch.tensor([np.pi/4, -np.pi/4, 3*np.pi/4, -3*np.pi/4])))

    def test_mode_shape_2_complex(self):
        # Test case with 2D motion spectrum
        mode_shape_2d = torch.tensor([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
            [[0, 0], [0, 0]]
        ]).unsqueeze(0)
        complex_tensor_2d = math_helper.mode_shape_2_complex(mode_shape_2d)
        expected_complex_tensor_2d = torch.tensor([
            [[1+5j, 2+6j], [3+7j, 4+8j]],
            [[9+0j, 10+0j], [11+0j, 12+0j]]
        ])
        self.assertTrue(torch.allclose(complex_tensor_2d, expected_complex_tensor_2d))
        self.assertEqual(complex_tensor_2d.shape[1], 2)

        # Test case with 3D motion spectrum
        mode_shape_3d = torch.tensor([
            [[1, 2], [3, 4]], 
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]], 
            [[13, 14], [15, 16]],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]]
        ]).unsqueeze(0)
        complex_tensor_3d = math_helper.mode_shape_2_complex(mode_shape_3d)
        expected_complex_tensor_3d = torch.tensor([
            [[1+5j, 2+6j], [3+7j, 4+8j]], 
            [[9+13j, 10+14j], [11+15j, 12+16j]],
            [[0+0j, 0+0j], [0+0j, 0+0j]]
        ])
        self.assertTrue(torch.allclose(complex_tensor_3d, expected_complex_tensor_3d))
        self.assertEqual(complex_tensor_3d.shape[1], 3)

        # Test case with 4D motion spectrum returns ValueError
        motion_spectrum_1d = torch.tensor([
            [[1, 2], [3, 4]], 
            [[5, 6], [7, 8]]
        ]).unsqueeze(0)
        with self.assertRaises(ValueError):
            math_helper.mode_shape_2_complex(motion_spectrum_1d)


    def test_normalize_modal_coordinate(self):
        # Test case 1
        modal_coordinate = torch.tensor([[1, 2], [3, 4]]).to(dtype=torch.float)
        normalized_modal_coordinate = math_helper.normalize_modal_coordinate(modal_coordinate)
        expected_normalized_modal_coordinate = torch.tensor([[0.4472, 0.8944], [0.6, 0.8]])
        self.assertTrue(torch.allclose(normalized_modal_coordinate, expected_normalized_modal_coordinate, atol=1e-6, rtol=1e-4))
        self.assertTrue(torch.allclose(torch.linalg.norm(normalized_modal_coordinate, dim=1, ord=2), torch.tensor(1.).expand_as(normalized_modal_coordinate[:, 0])))

        # Test case 2
        modal_coordinate = torch.tensor([[1, 1], [1, 1]]).to(dtype=torch.float)
        normalized_modal_coordinate = math_helper.normalize_modal_coordinate(modal_coordinate)
        expected_normalized_modal_coordinate = torch.tensor([[0.7071, 0.7071], [0.7071, 0.7071]])
        self.assertTrue(torch.allclose(normalized_modal_coordinate, expected_normalized_modal_coordinate, atol=1e-6, rtol=1e-4))
        self.assertTrue(torch.allclose(torch.linalg.norm(normalized_modal_coordinate, dim=1, ord=2), torch.tensor(1.).expand_as(normalized_modal_coordinate[:, 0])))
    
    def test_make_grid(self):
        # Test case 1: Single image
        img = torch.randn(1, 3, 256, 256)
        grid = math_helper.make_grid(img)
        self.assertEqual(grid.shape, (1, 2, 256, 256))
        self.assertTrue(torch.allclose(grid[:, 0], torch.arange(0, 256).view(1, 1, 256).float()))
        self.assertTrue(torch.allclose(grid[:, 1], torch.arange(0, 256).view(1, 256, 1).float()))

        # Test case 2: Multiple images
        img = torch.randn(2, 3, 128, 128)
        grid = math_helper.make_grid(img)
        self.assertEqual(grid.shape, (2, 2, 128, 128))
        self.assertTrue(torch.allclose(grid[:, 0], torch.arange(0, 128).view(1, 1, 128).float()))
        self.assertTrue(torch.allclose(grid[:, 1], torch.arange(0, 128).view(1, 128, 1).float()))

        # Test case 3: Empty image returns empty grid
        img = torch.empty(0, 3, 256, 256)
        grid = math_helper.make_grid(img)
        self.assertEqual(grid.shape, (0, 2, 256, 256))

        # Test case 4: Bad input type returns AttributeError from .to()
        img = np.random.randn(1, 3, 256, 256)
        with self.assertRaises(AttributeError):
            grid = math_helper.make_grid(img)

if __name__ == "__main__":
    unittest.main()