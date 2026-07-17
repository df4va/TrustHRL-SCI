from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ODEParameters:
    production: Tensor
    decay: Tensor
    interaction: Tensor
    half_saturation: Tensor
    hill_power: Tensor
    intervention: Tensor
    severity_knots: Tensor
    homeostatic: Tensor
    pathological: Tensor
    saddle: Tensor
    safety_ceiling: Tensor
    daily_dose_limit: Tensor

    @property
    def severities(self) -> int:
        return self.production.shape[0]

    @property
    def cytokines(self) -> int:
        return self.production.shape[1]

    @property
    def actions(self) -> int:
        return self.intervention.shape[-1]

    def validate(self) -> None:
        levels, cytokines = self.production.shape
        if self.decay.shape != (levels, cytokines):
            raise ValueError("decay shape mismatch")
        matrix_shape = (levels, cytokines, cytokines)
        if self.interaction.shape != matrix_shape:
            raise ValueError("interaction shape mismatch")
        if self.half_saturation.shape != matrix_shape:
            raise ValueError("half-saturation shape mismatch")
        if self.hill_power.shape != matrix_shape:
            raise ValueError("Hill power shape mismatch")
        if self.intervention.shape[:2] != (levels, cytokines):
            raise ValueError("intervention shape mismatch")
        if self.severity_knots.shape != (levels,):
            raise ValueError("severity knot shape mismatch")
        for equilibrium in (self.homeostatic, self.pathological, self.saddle):
            if equilibrium.shape != (levels, cytokines):
                raise ValueError("equilibrium shape mismatch")
        if self.safety_ceiling.shape != (cytokines,):
            raise ValueError("safety ceiling shape mismatch")
        if self.daily_dose_limit.shape != (self.actions,):
            raise ValueError("dose limit shape mismatch")
        if torch.any(self.production < 0) or torch.any(self.decay < 0):
            raise ValueError("rates must be nonnegative")
        if torch.any(self.half_saturation <= 0):
            raise ValueError("half-saturation constants must be positive")
        if torch.any(self.hill_power <= 0):
            raise ValueError("Hill powers must be positive")

    def to(self, device: torch.device, dtype: torch.dtype = torch.float32) -> ODEParameters:
        values = {
            name: value.to(device=device, dtype=dtype) for name, value in self.__dict__.items()
        }
        return ODEParameters(**values)

    @classmethod
    def load(cls, path: str | Path) -> ODEParameters:
        with np.load(Path(path), allow_pickle=False) as data:
            values = {name: torch.from_numpy(data[name]) for name in data.files}
        instance = cls(**values)
        instance.validate()
        return instance

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: value.detach().cpu().numpy() for name, value in self.__dict__.items()}
        np.savez_compressed(target, **arrays)


class SeverityInterpolator(nn.Module):
    def __init__(self, knots: Tensor) -> None:
        super().__init__()
        self.register_buffer("knots", knots)

    def forward(self, values: Tensor, severity: Tensor) -> Tensor:
        severity = severity.clamp(self.knots[0], self.knots[-1])
        upper = torch.searchsorted(self.knots, severity, right=True)
        upper = upper.clamp(1, self.knots.numel() - 1)
        lower = upper - 1
        low_knot = self.knots[lower]
        high_knot = self.knots[upper]
        fraction = (severity - low_knot) / (high_knot - low_knot).clamp_min(1e-12)
        low_value = values.index_select(0, lower.reshape(-1)).reshape(
            *severity.shape, *values.shape[1:]
        )
        high_value = values.index_select(0, upper.reshape(-1)).reshape(
            *severity.shape, *values.shape[1:]
        )
        while fraction.ndim < low_value.ndim:
            fraction = fraction.unsqueeze(-1)
        return low_value + fraction * (high_value - low_value)


class CytokineNetwork(nn.Module):
    def __init__(self, parameters: ODEParameters, kinetics: str = "hill") -> None:
        super().__init__()
        parameters.validate()
        self.cytokines = parameters.cytokines
        self.actions = parameters.actions
        self.kinetics = kinetics
        self.register_buffer("production", parameters.production)
        self.register_buffer("decay", parameters.decay)
        self.register_buffer("interaction", parameters.interaction)
        self.register_buffer("half_saturation", parameters.half_saturation)
        self.register_buffer("hill_power", parameters.hill_power)
        self.register_buffer("intervention", parameters.intervention)
        self.register_buffer("severity_knots", parameters.severity_knots)
        self.interpolator = SeverityInterpolator(parameters.severity_knots)

    def severity_parameters(self, severity: Tensor) -> Mapping[str, Tensor]:
        return {
            "production": self.interpolator(self.production, severity),
            "decay": self.interpolator(self.decay, severity),
            "interaction": self.interpolator(self.interaction, severity),
            "half_saturation": self.interpolator(self.half_saturation, severity),
            "hill_power": self.interpolator(self.hill_power, severity),
            "intervention": self.interpolator(self.intervention, severity),
        }

    def interactions(self, state: Tensor, values: Mapping[str, Tensor]) -> Tensor:
        source = state.unsqueeze(-2)
        if self.kinetics == "hill":
            powered_state = source.clamp_min(0).pow(values["hill_power"])
            powered_half = values["half_saturation"].pow(values["hill_power"])
            activation = powered_state / (powered_half + powered_state).clamp_min(1e-12)
        elif self.kinetics == "linear":
            activation = source / values["half_saturation"].clamp_min(1e-12)
        else:
            raise ValueError(f"unknown kinetics {self.kinetics}")
        diagonal = torch.eye(self.cytokines, device=state.device, dtype=torch.bool)
        activation = activation.masked_fill(diagonal, 0)
        return 1 + (values["interaction"] * activation).sum(dim=-1)

    def forward(self, state: Tensor, action: Tensor, severity: Tensor) -> Tensor:
        if state.shape[-1] != self.cytokines:
            raise ValueError("state width differs from network")
        if action.shape[-1] != self.actions:
            raise ValueError("action width differs from network")
        values = self.severity_parameters(severity)
        production = values["production"] * self.interactions(state, values)
        decay = values["decay"] * state
        control = torch.einsum("...ca,...a->...c", values["intervention"], action)
        return production - decay + control


class RK4Integrator(nn.Module):
    def __init__(self, network: CytokineNetwork, step_hours: float, substeps: int) -> None:
        super().__init__()
        if step_hours <= 0 or substeps <= 0:
            raise ValueError("integration increments must be positive")
        self.network = network
        self.step_hours = step_hours
        self.substeps = substeps

    def substep(self, state: Tensor, action: Tensor, severity: Tensor, dt: float) -> Tensor:
        k1 = self.network(state, action, severity)
        k2 = self.network(state + 0.5 * dt * k1, action, severity)
        k3 = self.network(state + 0.5 * dt * k2, action, severity)
        k4 = self.network(state + dt * k3, action, severity)
        update = dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return (state + update).clamp_min(0)

    def forward(self, state: Tensor, action: Tensor, severity: Tensor) -> Tensor:
        result = state
        dt = self.step_hours / self.substeps
        for _ in range(self.substeps):
            result = self.substep(result, action, severity, dt)
        return result
