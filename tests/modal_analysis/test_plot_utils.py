import unittest
import torch
from pathlib import Path
from pymodal_surgical.modal_analysis.plot_utils import plot_modal_coordinates
import shutil

script_dir = Path(__file__).resolve().parent

class TestPlotUtils(unittest.TestCase):
    
    def test_plot_modal_coordinates(self):

        save_dir = Path("./tests/figures")

        displacement = torch.tensor([1.5, 2.5])
        pixel = (10, 20)
        str_displacement = f"({displacement[0].item()}_{displacement[1].item()})"
        str_pixel = f"({pixel[0]}_{pixel[1]})" 
        expected_filename = save_dir/f"modal_coordinates_{str_displacement}_{str_pixel}.png"
        # Test case 1
        modal_coordinates = torch.tensor([[1, 2], [3, 4]]).to(dtype=torch.cfloat)
        
        plot_modal_coordinates(modal_coordinates, displacement, pixel, show=False, save=True, save_dir=save_dir)
        self.assertTrue(expected_filename.exists())


        # Test case 2
        displacement = torch.tensor([1.5, 2.5, -0.5])
        pixel = (10, 20)
        str_displacement = f"({displacement[0].item()}_{displacement[1].item()}_{displacement[2].item()})"
        str_pixel = f"({pixel[0]}_{pixel[1]})"
        
        modal_coordinates = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(dtype=torch.cfloat)
        expected_filename = save_dir/f"modal_coordinates_{str_displacement}_{str_pixel}.png"
        plot_modal_coordinates(modal_coordinates, displacement, pixel, show=False, save=True, save_dir=save_dir)
        self.assertTrue(expected_filename.exists())

        # Clean up the test directory
        shutil.rmtree(save_dir)

if __name__ == "__main__":
    unittest.main()