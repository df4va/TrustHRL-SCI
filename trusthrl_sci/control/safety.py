from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ConstraintReport:
    discounted_cost: Tensor
    budget: Tensor
    violation: Tensor
    satisfaction_rate: Tensor


class PhaseLagrangian(nn.Module):
    def __init__(
        self,
        phases: int = 3,
        constraints: int = 3,
        learning_rate: float = 0.01,
        initial_value: float = 0.0,
    ) -> None:
        super().__init__()
        if phases <= 0 or constraints <= 0:
            raise ValueError("dimensions must be positive")
        self.phases = phases
        self.constraints = constraints
        self.learning_rate = learning_rate
        initial = torch.full((constraints, phases), initial_value)
        self.register_buffer("multipliers", initial)

    def penalty(self, cost_advantages: Tensor, phase: Tensor) -> Tensor:
        if cost_advantages.shape[-1] != self.constraints:
            raise ValueError("constraint width mismatch")
        weights = self.multipliers.transpose(0, 1).index_select(0, phase.reshape(-1))
        weights = weights.reshape(*phase.shape, self.constraints)
        return (weights * cost_advantages).sum(dim=-1)

    @torch.no_grad()
    def update(self, phase_costs: Tensor, budgets: Tensor) -> Tensor:
        expected = (self.constraints, self.phases)
        if phase_costs.shape != expected or budgets.shape != expected:
            raise ValueError("phase cost shape mismatch")
        violation = phase_costs - budgets
        self.multipliers.add_(self.learning_rate * violation).clamp_(min=0)
        return self.multipliers.clone()

    def reports(self, phase_costs: Tensor, budgets: Tensor, counts: Tensor) -> ConstraintReport:
        violation = phase_costs - budgets
        satisfaction = 1 - phase_costs / counts.clamp_min(1)
        return ConstraintReport(phase_costs, budgets, violation, satisfaction)


def discounted_costs(costs: Tensor, terminated: Tensor, discount: float) -> Tensor:
    if costs.ndim != 3:
        raise ValueError("costs must have time, batch, and constraint axes")
    result = torch.zeros_like(costs)
    accumulator = torch.zeros_like(costs[-1])
    for index in range(costs.shape[0] - 1, -1, -1):
        continuation = 1 - terminated[index].float().unsqueeze(-1)
        accumulator = costs[index] + discount * continuation * accumulator
        result[index] = accumulator
    return result


def aggregate_phase_costs(costs: Tensor, phases: Tensor, phase_count: int = 3) -> Tensor:
    if costs.shape[:-1] != phases.shape:
        raise ValueError("phase index shape mismatch")
    constraints = costs.shape[-1]
    result = torch.zeros((constraints, phase_count), device=costs.device, dtype=costs.dtype)
    for phase in range(phase_count):
        mask = phases == phase
        if torch.any(mask):
            result[:, phase] = costs[mask].sum(dim=0)
    return result


def phase_budgets(
    phases: Tensor,
    budget_rate: float,
    constraints: int = 3,
    phase_count: int = 3,
) -> Tensor:
    counts = torch.stack([(phases == phase).sum() for phase in range(phase_count)]).float()
    return budget_rate * counts.unsqueeze(0).expand(constraints, -1)
