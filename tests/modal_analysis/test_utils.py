import unittest
import torch
from pathlib import Path
from pymodal_surgical.modal_analysis.utils import save_modal_coordinates, load_modal_coordinates
import shutil

script_dir = Path(__file__).resolve().parent

class TestUtils(unittest.TestCase):

    def test_load_save_modal_coordinates(self):
        # Test case with displacement and pixel coordinates
        modal_coordinates = torch.tensor([[1, 2], [3, 4]])
        displacement = torch.tensor([1.5, 2.5])
        pixel = (10, 20)
        save_dir = script_dir/"test_data"
        expected_filename = save_dir/"modal_coordinates/disp_1.5_2.5_px_10_20.npy"

        # Call the function
        save_modal_coordinates(modal_coordinates, save_dir, displacement, pixel)

        # Check if the file exists
        self.assertTrue(Path(expected_filename).exists())

        # Check that the modal coordinates are the same
        loaded_modal_coordinates = load_modal_coordinates(expected_filename)
        self.assertTrue(torch.allclose(modal_coordinates, loaded_modal_coordinates))

        # Clean up the test directory
        shutil.rmtree(save_dir/"modal_coordinates")
    
    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(script_dir/"test_data")

if __name__ == "__main__":
    unittest.main()