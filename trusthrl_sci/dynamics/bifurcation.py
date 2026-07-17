from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from trusthrl_sci.dynamics.network import CytokineNetwork, ODEParameters, RK4Integrator


@dataclass(frozen=True)
class FixedPoint:
    state: Tensor
    eigenvalues: Tensor
    stable: bool
    severity: float


@dataclass(frozen=True)
class BifurcationSlice:
    severity: float
    fixed_points: tuple[FixedPoint, ...]


def jacobian(network: CytokineNetwork, state: Tensor, severity: float) -> Tensor:
    single_state = state.detach().requires_grad_(True)
    action = torch.zeros(network.actions, dtype=single_state.dtype, device=single_state.device)
    severity_tensor = torch.tensor(severity, dtype=single_state.dtype, device=single_state.device)

    def function(value: Tensor) -> Tensor:
        return network(value, action, severity_tensor)

    return torch.autograd.functional.jacobian(function, single_state)


def classify(network: CytokineNetwork, state: Tensor, severity: float) -> FixedPoint:
    matrix = jacobian(network, state, severity)
    eigenvalues = torch.linalg.eigvals(matrix)
    stable = bool(torch.all(eigenvalues.real < 0).item())
    return FixedPoint(state.detach(), eigenvalues.detach(), stable, severity)


def converge(
    network: CytokineNetwork,
    starts: Tensor,
    severity: float,
    step_hours: float = 0.1,
    max_steps: int = 20000,
    tolerance: float = 1e-7,
) -> Tensor:
    integrator = RK4Integrator(network, step_hours, 1)
    action = torch.zeros((starts.shape[0], network.actions), device=starts.device)
    severity_tensor = torch.full((starts.shape[0],), severity, device=starts.device)
    state = starts.clone()
    for _ in range(max_steps):
        next_state = integrator(state, action, severity_tensor)
        difference = torch.linalg.vector_norm(next_state - state, dim=-1)
        state = next_state
        if torch.all(difference < tolerance):
            break
    return state


def deduplicate(points: Tensor, tolerance: float = 1e-4) -> Tensor:
    accepted: list[Tensor] = []
    for point in points:
        if not accepted:
            accepted.append(point)
            continue
        distances = torch.stack([torch.linalg.vector_norm(point - item) for item in accepted])
        if torch.all(distances > tolerance):
            accepted.append(point)
    if not accepted:
        return points[:0]
    return torch.stack(accepted)


def trace_severity(
    parameters: ODEParameters,
    starts: Tensor,
    severities: Iterable[float],
) -> tuple[BifurcationSlice, ...]:
    network = CytokineNetwork(parameters)
    slices = []
    current_starts = starts
    for severity in severities:
        converged = deduplicate(converge(network, current_starts, severity))
        fixed_points = tuple(classify(network, point, severity) for point in converged)
        slices.append(BifurcationSlice(severity, fixed_points))
        current_starts = torch.cat((starts, converged), dim=0)
    return tuple(slices)


def basin_membership(
    state: Tensor,
    homeostatic: Tensor,
    pathological: Tensor,
    saddle: Tensor,
) -> Tensor:
    homeostatic_distance = torch.linalg.vector_norm(state - homeostatic, dim=-1)
    pathological_distance = torch.linalg.vector_norm(state - pathological, dim=-1)
    saddle_radius = torch.linalg.vector_norm(saddle - homeostatic, dim=-1)
    inside_radius = homeostatic_distance <= saddle_radius
    closer = homeostatic_distance < pathological_distance
    return inside_radius & closer
