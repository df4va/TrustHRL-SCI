from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import torch
from torch import Tensor


class Phase(IntEnum):
    ACUTE = 0
    SUBACUTE = 1
    CHRONIC = 2


class Constraint(IntEnum):
    CONCENTRATION = 0
    DOSE = 1
    IMMUNODEFICIENCY = 2


class PolicyOutput(NamedTuple):
    action: Tensor
    log_probability: Tensor
    entropy: Tensor
    value: Tensor


class StepOutput(NamedTuple):
    state: Tensor
    reward: Tensor
    costs: Tensor
    terminated: Tensor


@dataclass(frozen=True)
class Equilibria:
    homeostatic: Tensor
    pathological: Tensor
    saddle: Tensor

    def validate(self, cytokines: int) -> None:
        expected = (cytokines,)
        if self.homeostatic.shape != expected:
            raise ValueError("homeostatic equilibrium has incorrect shape")
        if self.pathological.shape != expected:
            raise ValueError("pathological equilibrium has incorrect shape")
        if self.saddle.shape != expected:
            raise ValueError("saddle equilibrium has incorrect shape")
        if torch.any(self.homeostatic < 0):
            raise ValueError("homeostatic equilibrium must be nonnegative")


@dataclass(frozen=True)
class Observation:
    cytokines: Tensor
    severity: Tensor
    time: Tensor
    phase_estimate: Tensor

    def low_level(self, include_severity: bool = True) -> Tensor:
        if include_severity:
            return torch.cat((self.cytokines, self.severity.unsqueeze(-1)), dim=-1)
        return self.cytokines

    def high_level(self, include_severity: bool = True) -> Tensor:
        base = self.low_level(include_severity)
        return torch.cat((base, self.time.unsqueeze(-1), self.phase_estimate.unsqueeze(-1)), dim=-1)


@dataclass(frozen=True)
class Transition:
    observation: Tensor
    action: Tensor
    log_probability: Tensor
    reward: Tensor
    costs: Tensor
    value: Tensor
    terminated: Tensor
    phase: Tensor


@dataclass(frozen=True)
class RolloutBatch:
    observations: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    advantages: Tensor
    returns: Tensor
    cost_advantages: Tensor
    phases: Tensor

    def size(self) -> int:
        return self.observations.shape[0]

    def to(self, device: torch.device) -> RolloutBatch:
        return RolloutBatch(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            old_log_probabilities=self.old_log_probabilities.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
            cost_advantages=self.cost_advantages.to(device),
            phases=self.phases.to(device),
        )
