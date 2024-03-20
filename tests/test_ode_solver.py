import ode_solver 
import unittest
import numpy as np
import torch

class TestSolver(unittest.TestCase):
    
    def test_solver_solutions(self):
        # Test case 1
        modal_coordinate = torch.tensor([[1+2j, 2+3j]])
        frequencies = torch.tensor([1])
        modal_mass = torch.tensor([1])
        force = torch.tensor([[1., 2.]]).to(dtype=torch.cfloat)
        alpha = 0.1
        beta = 0.1
        time_step = 0.1

        result = ode_solver.euler_solver(modal_coordinate, frequencies, modal_mass, force, alpha, beta, time_step)
        expected_result = torch.tensor([[1.0000+1.8600j, 1.9000+2.9400j]])
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-4, rtol=1e-4))

        # Test case 2
        modal_coordinate = torch.tensor([[1-2j, 2-3j]])
        frequencies = torch.tensor([1])
        modal_mass = torch.tensor([1])
        force = torch.tensor([[-1., -2.]]).to(dtype=torch.cfloat)
        alpha = 0.2
        beta = 0.2
        time_step = 0.2

        result = ode_solver.euler_solver(modal_coordinate, frequencies, modal_mass, force, alpha, beta, time_step)
        expected_result = torch.tensor([[1.0000-1.2400j, 2.2000-1.9600j]])
        self.assertTrue(torch.allclose(result, expected_result))

        # Test case 3
        modal_coordinate = torch.tensor([[1+1j, 2+2j], [2+2j, 3+3j]])
        frequencies = torch.tensor([1, 2])
        modal_mass = torch.tensor([1, 2])
        force = torch.tensor([[1., 2.], [0.5, -1.0]]).to(dtype=torch.cfloat)
        alpha = 0.1
        beta = 0.1
        time_step = 0.05

        
        result = ode_solver.euler_solver(modal_coordinate, frequencies, modal_mass, force, alpha, beta, time_step)
        expected_result = torch.tensor([
            [1.0500+0.9400j, 2.0000+1.9800j],
            [1.7750+2.1625j, 2.6750+3.2375j]
        ])

        self.assertTrue(torch.allclose(result, expected_result, atol=1e-4, rtol=1e-4))
    
    def test_solver_data_type(self):

        # Test case 1
        modal_coordinates = np.random.randn(100, 2, 2)
        mass = np.random.randint(1, 10, 100)
        frequencies = np.random.randint(1, 10, 100)
        force = np.random.randn(2, 2)
        alpha = 0.1
        beta = 0.1
        time_step = 0.1

        with self.assertRaises(AttributeError):
            result = ode_solver.euler_solver(modal_coordinates, frequencies, mass, force, alpha, beta, time_step)
        
        # Test case 2
        modal_coordinate = torch.tensor([[1+2j, 2+3j]])
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