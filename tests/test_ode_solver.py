import ode_solver 
import unittest
import numpy as np
import torch

class TestSolver(unittest.TestCase):
    
    def test_solver_solutions(self):
        # Test case 1
        modal_coordinate = torch.tensor([[1+2j, 2+3j, 3+4j, 4+5j]])
        frequencies = torch.tensor([1])
        modal_mass = torch.tensor([1])
        force = torch.tensor([[1., 2.]]).to(dtype=torch.cfloat)
        alpha = 0.1
        beta = 0.1
        time_step = 0.1

        result = ode_solver.euler_solver(modal_coordinate, frequencies, modal_mass, force, alpha, beta, time_step)
        expected_result = torch.tensor([[1.0 + 1.86j, 1.9 + 2.94j, 2.8 + 4.02j, 3.7 + 5.1j]])
        self.assertTrue(torch.allclose(result, expected_result))

        # Test case 2
        modal_coordinate = torch.tensor([[1-2j, 2-3j, 3-4j, 4-5j]])
        frequencies = torch.tensor([1])
        modal_mass = torch.tensor([1])
        force = torch.tensor([[-1., -2.]]).to(dtype=torch.cfloat)
        alpha = 0.2
        beta = 0.2
        time_step = 0.2

        result = ode_solver.euler_solver(modal_coordinate, frequencies, modal_mass, force, alpha, beta, time_step)
        expected_result = torch.tensor([[1.0-1.24j, 2.2-1.96j, 3.4-2.68j, 4.6-3.4j]])
        self.assertTrue(torch.allclose(result, expected_result))

        # Test case 3
        modal_coordinate = torch.tensor([[1+1j, 2+2j, 3+3j, 4+4j], [2+2j, 3+3j, 4+4j, 5+5j]])
        frequencies = torch.tensor([1, 2])
        modal_mass = torch.tensor([1, 2])
        force = torch.tensor([[1., 2.], [0.5, -1.0]]).to(dtype=torch.cfloat)
        alpha = 0.1
        beta = 0.1
        time_step = 0.05

        
        result = ode_solver.euler_solver(modal_coordinate, frequencies, modal_mass, force, alpha, beta, time_step)
        expected_result = torch.tensor([
            [1.05+0.94j, 2.0+1.98j, 2.95+3.02j, 3.9+4.06j], 
            [1.925+0.7063j, 2.9+1.0531j, 3.875+1.4j, 4.85+1.7469j]
        ])

        self.assertTrue(torch.allclose(result, expected_result, atol=1e-4, rtol=1e-4))
    
    def test_solver_data_type(self):

        # Test case 1
        modal_coordinates = np.random.randn(100, 4, 2)
        mass = np.random.randint(1, 10, 100)
        frequencies = np.random.randint(1, 10, 100)
        force = np.random.randn(4, 2)
        alpha = 0.1
        beta = 0.1
        time_step = 0.1

        with self.assertRaises(AttributeError):
            result = ode_solver.euler_solver(modal_coordinates, frequencies, mass, force, alpha, beta, time_step)
        
        # Test case 2
        modal_coordinate = torch.tensor([[1+2j, 2+3j, 3+4j, 4+5j]])
        frequencies = torch.tensor([1])
        modal_mass = torch.tensor([1])
        force = torch.tensor([[1., 2.]]).to(dtype=torch.cfloat)
        alpha = 0.1
        beta = 0.1
        time_step = 0.1

        result = ode_solver.euler_solver(modal_coordinate, frequencies, modal_mass, force, alpha, beta, time_step)
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(result.dtype == torch.cfloat)
        self.assertTrue(result.shape == modal_coordinate.shape)