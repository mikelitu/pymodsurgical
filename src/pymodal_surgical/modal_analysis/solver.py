import torch

def euler_solver(
    modal_coordinate: torch.Tensor,
    frequencies: torch.Tensor,
    modal_mass: torch.Tensor,
    force: torch.Tensor,
    alpha: float = 0.1,
    beta: float = 0.1,
    time_step: float = 0.1,
) -> torch.Tensor:
    """
    Solves the modal analysis problem using the Euler method.

    Args:
        modal_coordinate (torch.Tensor): The modal coordinates of the system.
        frequencies (torch.Tensor): The natural frequencies of the system.
        modal_mass (torch.Tensor): The modal masses of the system.
        force (torch.Tensor): The external forces applied to the system.
        alpha (float, optional): The damping coefficient. Defaults to 0.1.
        beta (float, optional): The stiffness coefficient. Defaults to 0.1.
        time_step (float, optional): The time step for the simulation. Defaults to 0.1.

    Returns:
        torch.Tensor: The updated modal coordinates after the simulation.

    """
    modal_displacements = modal_coordinate.real
    modal_velocities = - modal_coordinate.imag * frequencies.unsqueeze(-1)
    new_q = []

    for md, mv, omega, m, f in zip(modal_displacements, modal_velocities, frequencies, modal_mass, force):
        y = torch.cat([md.unsqueeze(0), mv.unsqueeze(0)]).to(dtype=torch.cfloat, device=f.device)
        xi = (1/2) * ((alpha / omega) + beta * omega)
        A = torch.tensor([[1, time_step], [-omega**2 * time_step, 1 - 2 * xi * omega * time_step]]).to(dtype=torch.cfloat, device=y.device)
        B = torch.tensor([[0], [time_step / m]]).squeeze(1).to(dtype=torch.cfloat).to(device=y.device)
        tmp_y = A @ y + B @ f
        tmp_q = tmp_y[0, :] - 1j * (tmp_y[1, :] / omega)        
        new_q.append(tmp_q)
    
    return torch.stack(new_q).to(dtype=torch.cfloat, device=modal_coordinate.device)
