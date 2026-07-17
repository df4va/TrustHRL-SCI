from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class Controller(Protocol):
    def reset(self, batch_size: int, device: torch.device) -> None: ...

    def action(self, state: Tensor, target: Tensor, time: Tensor) -> Tensor: ...


class RandomController:
    def __init__(self, actions: int, seed: int = 17) -> None:
        self.actions = actions
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def reset(self, batch_size: int, device: torch.device) -> None:
        self.batch_size = batch_size
        self.device = device

    def action(self, state: Tensor, target: Tensor, time: Tensor) -> Tensor:
        return torch.rand(
            (state.shape[0], self.actions),
            generator=self.generator,
            device=state.device,
        )


@dataclass(frozen=True)
class PIDGains:
    proportional: Tensor
    integral: Tensor
    derivative: Tensor


class PIDController:
    def __init__(self, gains: PIDGains, mapping: Tensor, step_hours: float = 1.0) -> None:
        self.gains = gains
        self.mapping = mapping
        self.step_hours = step_hours
        self.integral: Tensor | None = None
        self.previous_error: Tensor | None = None

    def reset(self, batch_size: int, device: torch.device) -> None:
        cytokines = self.mapping.shape[1]
        self.integral = torch.zeros((batch_size, cytokines), device=device)
        self.previous_error = torch.zeros((batch_size, cytokines), device=device)

    def action(self, state: Tensor, target: Tensor, time: Tensor) -> Tensor:
        if self.integral is None or self.previous_error is None:
            raise RuntimeError("controller must be reset")
        error = target - state
        self.integral += error * self.step_hours
        derivative = (error - self.previous_error) / self.step_hours
        signal = (
            self.gains.proportional * error
            + self.gains.integral * self.integral
            + self.gains.derivative * derivative
        )
        self.previous_error.copy_(error)
        return torch.sigmoid(torch.einsum("ac,bc->ba", self.mapping, signal))


@dataclass(frozen=True)
class DoseEvent:
    start_hour: float
    end_hour: float
    dose: tuple[float, ...]


class FixedProtocol:
    def __init__(self, actions: int, events: Sequence[DoseEvent]) -> None:
        self.actions = actions
        self.events = tuple(events)
        for event in self.events:
            if len(event.dose) != actions:
                raise ValueError("event dose width mismatch")
            if event.end_hour <= event.start_hour:
                raise ValueError("event must have positive duration")

    def reset(self, batch_size: int, device: torch.device) -> None:
        self.batch_size = batch_size
        self.device = device

    def action(self, state: Tensor, target: Tensor, time: Tensor) -> Tensor:
        result = torch.zeros((state.shape[0], self.actions), device=state.device)
        for event in self.events:
            mask = (time >= event.start_hour) & (time < event.end_hour)
            dose = torch.tensor(event.dose, device=state.device)
            result[mask] = dose
        return result.clamp(0, 1)


class ZeroController:
    def __init__(self, actions: int) -> None:
        self.actions = actions

    def reset(self, batch_size: int, device: torch.device) -> None:
        self.batch_size = batch_size
        self.device = device

    def action(self, state: Tensor, target: Tensor, time: Tensor) -> Tensor:
        return torch.zeros((state.shape[0], self.actions), device=state.device)


def ziegler_nichols(ultimate_gain: Tensor, ultimate_period: Tensor) -> PIDGains:
    proportional = 0.6 * ultimate_gain
    integral = 1.2 * ultimate_gain / ultimate_period.clamp_min(1e-12)
    derivative = 0.075 * ultimate_gain * ultimate_period
    return PIDGains(proportional, integral, derivative)
