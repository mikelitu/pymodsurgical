import unittest
import torch
import numpy as np
from pathlib import Path
import pymodal_surgical.complex 
import shutil

class TestComplex(unittest.TestCase):
    def test_tensor_rotate_phase_to_real_axis(self):
        tensor = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j])
        real_axis = pymodal_surgical.complex.tensor_rotate_phase_to_real_axis(tensor)
        self.assertTrue(torch.allclose(real_axis, torch.tensor([1.4142, 1.4142, 1.4142, 1.4142])))

    def test_complex_from_magnitude_phase(self):
        magnitude = torch.tensor([1.4142, 1.4142, 1.4142, 1.4142])
        phase = torch.tensor([np.pi/4, -np.pi/4, 3*np.pi/4, -3*np.pi/4])
        complex_tensor = pymodal_surgical.complex.complex_from_magnitude_phase(magnitude, phase)
        self.assertTrue(torch.allclose(complex_tensor, torch.tensor([1+1j, 1-1j, -1+1j, -1-1j])))

    def test_complex_to_magnitude_phase(self):
        complex_tensor = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j])
        magnitude, phase = pymodal_surgical.complex.complex_to_magnitude_phase(complex_tensor)
        self.assertTrue(torch.allclose(magnitude, torch.tensor([1.4142, 1.4142, 1.4142, 1.4142])))
        self.assertTrue(torch.allclose(phase, torch.tensor([np.pi/4, -np.pi/4, 3*np.pi/4, -3*np.pi/4])))

    def test_motion_spectrum_2_complex(self):
        # Test case with 2D motion spectrum
        motion_spectrum_2d = torch.tensor([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
            [[0, 0], [0, 0]]
        ]).unsqueeze(0)
        complex_tensor_2d = pymodal_surgical.complex.motion_spectrum_2_complex(motion_spectrum_2d)
        expected_complex_tensor_2d = torch.tensor([
            [[1+5j, 2+6j], [3+7j, 4+8j]],
            [[9+0j, 10+0j], [11+0j, 12+0j]]
        ])
        self.assertTrue(torch.allclose(complex_tensor_2d, expected_complex_tensor_2d))
        self.assertEqual(complex_tensor_2d.shape[1], 2)

        # Test case with 3D motion spectrum
        motion_spectrum_3d = torch.tensor([
            [[1, 2], [3, 4]], 
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]], 
            [[13, 14], [15, 16]],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]]
        ]).unsqueeze(0)
        complex_tensor_3d = pymodal_surgical.complex.motion_spectrum_2_complex(motion_spectrum_3d)
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
            pymodal_surgical.complex.motion_spectrum_2_complex(motion_spectrum_1d)
    

    def test_load_save_modal_coordinates(self):
        # Test case with displacement and pixel coordinates
        modal_coordinates = torch.tensor([[1, 2], [3, 4]])
        displacement = torch.tensor([1.5, 2.5])
        pixel = (10, 20)
        save_dir = Path("./tests")
        expected_filename = save_dir/"modal_coordinates/disp_1.5_2.5_px_10_20.npy"

        # Call the function
        pymodal_surgical.complex.save_modal_coordinates(modal_coordinates, save_dir, displacement, pixel)

        # Check if the file exists
        self.assertTrue(Path(expected_filename).exists())

        # Check that the modal coordinates are the same
        loaded_modal_coordinates = pymodal_surgical.complex.load_modal_coordinates(expected_filename)
        self.assertTrue(torch.allclose(modal_coordinates, loaded_modal_coordinates))

        # Clean up the test directory
        shutil.rmtree(save_dir/"modal_coordinates")


    def test_normalize_modal_coordinate(self):
        # Test case 1
        modal_coordinate = torch.tensor([[1, 2], [3, 4]]).to(dtype=torch.float)
        normalized_modal_coordinate = pymodal_surgical.complex.normalize_modal_coordinate(modal_coordinate)
        expected_normalized_modal_coordinate = torch.tensor([[0.4472, 0.8944], [0.6, 0.8]])
        self.assertTrue(torch.allclose(normalized_modal_coordinate, expected_normalized_modal_coordinate, atol=1e-6, rtol=1e-4))
        self.assertTrue(torch.allclose(torch.linalg.norm(normalized_modal_coordinate, dim=1, ord=2), torch.tensor(1.).expand_as(normalized_modal_coordinate[:, 0])))

        # Test case 2
        modal_coordinate = torch.tensor([[1, 1], [1, 1]]).to(dtype=torch.float)
        normalized_modal_coordinate = pymodal_surgical.complex.normalize_modal_coordinate(modal_coordinate)
        expected_normalized_modal_coordinate = torch.tensor([[0.7071, 0.7071], [0.7071, 0.7071]])
        self.assertTrue(torch.allclose(normalized_modal_coordinate, expected_normalized_modal_coordinate, atol=1e-6, rtol=1e-4))
        self.assertTrue(torch.allclose(torch.linalg.norm(normalized_modal_coordinate, dim=1, ord=2), torch.tensor(1.).expand_as(normalized_modal_coordinate[:, 0])))


    def test_plot_modal_coordinates(self):

        save_dir = Path("./tests/figures")

        displacement = torch.tensor([1.5, 2.5])
        pixel = (10, 20)
        str_displacement = f"({displacement[0].item()}_{displacement[1].item()})"
        str_pixel = f"({pixel[0]}_{pixel[1]})" 
        expected_filename = save_dir/f"modal_coordinates_{str_displacement}_{str_pixel}.png"
        # Test case 1
        modal_coordinates = torch.tensor([[1, 2], [3, 4]]).to(dtype=torch.cfloat)
        
        pymodal_surgical.complex.plot_modal_coordinates(modal_coordinates, displacement, pixel, show=False, save=True, save_dir=save_dir)
        self.assertTrue(expected_filename.exists())


        # Test case 2
        displacement = torch.tensor([1.5, 2.5, -0.5])
        pixel = (10, 20)
        str_displacement = f"({displacement[0].item()}_{displacement[1].item()}_{displacement[2].item()})"
        str_pixel = f"({pixel[0]}_{pixel[1]})"
        
        modal_coordinates = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(dtype=torch.cfloat)
        expected_filename = save_dir/f"modal_coordinates_{str_displacement}_{str_pixel}.png"
        pymodal_surgical.complex.plot_modal_coordinates(modal_coordinates, displacement, pixel, show=False, save=True, save_dir=save_dir)
        self.assertTrue(expected_filename.exists())

        # Clean up the test directory
        shutil.rmtree(save_dir)

if __name__ == "__main__":
    unittest.main()