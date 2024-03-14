import numpy as np
import torch


def euler_solver(
    modal_coordinate: torch.Tensor,
    prev_modal_coordinate: torch.Tensor,
    frequencies: torch.Tensor,
    modal_mass: torch.Tensor,
    alpha: float,
    beta: float,
    time_step: float = 0.1,
) -> torch.Tensor:
    modal_displacements = modal_coordinate - prev_modal_coordinate
    modal_velocities = modal_displacements / time_step
    new_q = []

    for md, mv, omega, m in zip(modal_displacements, modal_velocities, frequencies, modal_mass):
        y = torch.concatenate((md.unsqueeze(0), mv.unsqueeze(0)))
        print(y)
        xi = (1/2) *((alpha / omega) + beta * omega)
        mv = (mv + xi * md) / (1 + xi * time_step)
        A = torch.tensor([[1, time_step], [-omega**2 * time_step, 1 - 2 * xi * time_step]])
        B = torch.tensor([[0], [time_step / m]])
        tmp_y = A @ y + B
        tmp_q = tmp_y[0, :] - 1j * (tmp_y[1, :] / omega)        
        new_q.append(tmp_q)
    
    return torch.stack(new_q)

if __name__ == "__main__":
    modal_coordinate = torch.tensor([[1, 2], [3, 4]]).to(dtype=torch.float)
    prev_modal_coordinate = torch.tensor([[1, 1], [1, 1]]).to(dtype=torch.float)
    frequencies = torch.tensor([0.1, 0.2]).to(dtype=torch.float)
    modal_mass = torch.tensor([1, 2]).to(dtype=torch.float)
    alpha = 0.01
    beta = 0.01
    time_step = 0.05
    new_modal_coordinate = euler_solver(modal_coordinate, prev_modal_coordinate, frequencies, modal_mass, alpha, beta, time_step)
    print(new_modal_coordinate)
    # tensor([[0.9000, 1.8000],
    #         [2.8000, 3.6000]])

