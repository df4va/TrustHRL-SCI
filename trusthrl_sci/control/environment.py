from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from trusthrl_sci.contracts import Observation, Phase, StepOutput
from trusthrl_sci.dynamics.catalog import pro_inflammatory_indices
from trusthrl_sci.dynamics.network import ODEParameters, RK4Integrator


@dataclass(frozen=True)
class EnvironmentLimits:
    concentration_ceiling: Tensor
    daily_dose_limit: Tensor
    immunodeficiency_floor: float
    safety_budget: float


class DoseWindow:
    def __init__(self, batch_size: int, actions: int, window: int, device: torch.device) -> None:
        self.window = window
        self.values = torch.zeros((batch_size, window, actions), device=device)
        self.cursor = 0
        self.filled = 0

    def append(self, action: Tensor) -> None:
        self.values[:, self.cursor] = action
        self.cursor = (self.cursor + 1) % self.window
        self.filled = min(self.filled + 1, self.window)

    def total(self) -> Tensor:
        return self.values[:, : self.filled].sum(dim=1)

    def reset(self, mask: Tensor | None = None) -> None:
        if mask is None:
            self.values.zero_()
            self.cursor = 0
            self.filled = 0
        else:
            self.values[mask] = 0


class InflammationEnvironment:
    def __init__(
        self,
        integrator: RK4Integrator,
        parameters: ODEParameters,
        batch_size: int,
        horizon: int = 168,
        phase_boundaries: tuple[int, ...] = (0, 6, 72, 168),
        immunodeficiency_floor: float = 0.05,
        reward_threshold: float = 0.8,
        device: torch.device | None = None,
    ) -> None:
        self.integrator = integrator
        self.parameters = parameters
        self.batch_size = batch_size
        self.horizon = horizon
        self.phase_boundaries = phase_boundaries
        self.immunodeficiency_floor = immunodeficiency_floor
        self.reward_threshold = reward_threshold
        self.device = device or torch.device("cpu")
        self.pro_indices = torch.tensor(pro_inflammatory_indices(), device=self.device)
        self.state = torch.empty((batch_size, parameters.cytokines), device=self.device)
        self.severity = torch.empty(batch_size, device=self.device)
        self.initial = torch.empty_like(self.state)
        self.homeostatic = torch.empty_like(self.state)
        self.pathological = torch.empty_like(self.state)
        self.saddle = torch.empty_like(self.state)
        self.time = torch.zeros(batch_size, device=self.device)
        self.phase = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.doses = DoseWindow(batch_size, parameters.actions, 24, self.device)
        self._initialized = False

    def _interpolate(self, values: Tensor, severity: Tensor) -> Tensor:
        knots = self.parameters.severity_knots
        upper = torch.searchsorted(knots, severity, right=True).clamp(1, knots.numel() - 1)
        lower = upper - 1
        fraction = (severity - knots[lower]) / (knots[upper] - knots[lower]).clamp_min(1e-12)
        low_value = values[lower]
        high_value = values[upper]
        while fraction.ndim < low_value.ndim:
            fraction = fraction.unsqueeze(-1)
        return low_value + fraction * (high_value - low_value)

    def endogenous_phase(self, time: Tensor) -> Tensor:
        result = torch.zeros_like(time, dtype=torch.long)
        for phase, boundary in enumerate(self.phase_boundaries[1:-1], start=1):
            result = torch.where(time >= boundary, phase, result)
        return result

    def reset(self, severity: Tensor | None = None) -> Observation:
        if severity is None:
            severity = torch.rand(self.batch_size, device=self.device)
        if severity.shape != (self.batch_size,):
            raise ValueError("severity shape mismatch")
        self.severity.copy_(severity)
        self.homeostatic.copy_(self._interpolate(self.parameters.homeostatic, severity))
        self.pathological.copy_(self._interpolate(self.parameters.pathological, severity))
        self.saddle.copy_(self._interpolate(self.parameters.saddle, severity))
        self.initial.copy_(self.pathological)
        self.state.copy_(self.initial)
        self.time.zero_()
        self.phase.zero_()
        self.doses.reset()
        self._initialized = True
        return self.observation()

    def observation(self) -> Observation:
        if not self._initialized:
            raise RuntimeError("environment must be reset")
        return Observation(
            cytokines=self.state.clone(),
            severity=self.severity.clone(),
            time=self.time.clone() / self.horizon,
            phase_estimate=self.endogenous_phase(self.time).float()
            / max(len(self.phase_boundaries) - 2, 1),
        )

    def set_phase(self, proposal: Tensor) -> Tensor:
        if proposal.shape != self.phase.shape:
            raise ValueError("phase proposal shape mismatch")
        proposal = proposal.clamp(int(Phase.ACUTE), len(self.phase_boundaries) - 2)
        self.phase = torch.maximum(self.phase, proposal)
        return self.phase.clone()

    def reward(self, state: Tensor) -> Tensor:
        numerator = torch.linalg.vector_norm(state - self.homeostatic, dim=-1)
        denominator = torch.linalg.vector_norm(self.initial - self.homeostatic, dim=-1)
        return 1 - numerator / denominator.clamp_min(1e-12)

    def basin_membership(self, state: Tensor) -> Tensor:
        home_distance = torch.linalg.vector_norm(state - self.homeostatic, dim=-1)
        path_distance = torch.linalg.vector_norm(state - self.pathological, dim=-1)
        saddle_distance = torch.linalg.vector_norm(self.saddle - self.homeostatic, dim=-1)
        return (home_distance <= saddle_distance) & (home_distance < path_distance)

    def costs(self, next_state: Tensor, action: Tensor) -> Tensor:
        concentration = torch.any(next_state > self.parameters.safety_ceiling.unsqueeze(0), dim=-1)
        self.doses.append(action)
        dose = torch.any(self.doses.total() > self.parameters.daily_dose_limit.unsqueeze(0), dim=-1)
        home_pro = self.homeostatic.index_select(-1, self.pro_indices)
        state_pro = next_state.index_select(-1, self.pro_indices)
        immunodeficiency = torch.any(state_pro < self.immunodeficiency_floor * home_pro, dim=-1)
        return torch.stack((concentration, dose, immunodeficiency), dim=-1).float()

    def step(self, action: Tensor) -> StepOutput:
        if not self._initialized:
            raise RuntimeError("environment must be reset")
        bounded_action = action.clamp(0, 1)
        next_state = self.integrator(self.state, bounded_action, self.severity)
        costs = self.costs(next_state, bounded_action)
        reward = self.reward(next_state)
        self.state.copy_(next_state)
        self.time.add_(1)
        terminated = self.time >= self.horizon
        return StepOutput(next_state.clone(), reward, costs, terminated)


class VectorEpisodeLedger:
    def __init__(self, batch_size: int, device: torch.device) -> None:
        self.rewards = torch.zeros(batch_size, device=device)
        self.steps = torch.zeros(batch_size, device=device)
        self.costs = torch.zeros((batch_size, 3), device=device)
        self.dose = torch.zeros(batch_size, device=device)
        self.homeostatic_steps = torch.zeros(batch_size, device=device)

    def update(
        self,
        reward: Tensor,
        costs: Tensor,
        action: Tensor,
        homeostatic: Tensor,
    ) -> None:
        self.rewards += reward
        self.steps += 1
        self.costs += costs
        self.dose += action.sum(dim=-1)
        self.homeostatic_steps += homeostatic.float()

    def summaries(self) -> dict[str, Tensor]:
        denominator = self.steps.clamp_min(1)
        return {
            "mean_reward": self.rewards / denominator,
            "hrs": 100 * self.homeostatic_steps / denominator,
            "scr": 100 * (1 - self.costs.sum(dim=-1) / (3 * denominator)),
            "efficiency": self.rewards / self.dose.clamp_min(1e-12),
            "concentration_violation": self.costs[:, 0] / denominator,
            "dose_violation": self.costs[:, 1] / denominator,
            "immunodeficiency_violation": self.costs[:, 2] / denominator,
        }
