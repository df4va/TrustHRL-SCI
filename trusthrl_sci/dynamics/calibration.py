from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from torch import Tensor

from trusthrl_sci.dynamics.network import CytokineNetwork, ODEParameters, RK4Integrator


@dataclass(frozen=True)
class CalibrationSeries:
    severity: float
    times: np.ndarray
    observations: np.ndarray
    mask: np.ndarray

    def validate(self, cytokines: int) -> None:
        if self.times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        if self.observations.shape != (self.times.size, cytokines):
            raise ValueError("observation shape mismatch")
        if self.mask.shape != self.observations.shape:
            raise ValueError("mask shape mismatch")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("times must increase")
        if not 0 <= self.severity <= 1:
            raise ValueError("severity must lie in [0, 1]")


@dataclass(frozen=True)
class CalibrationBounds:
    production: tuple[float, float]
    decay: tuple[float, float]
    interaction: tuple[float, float]
    half_saturation: tuple[float, float]
    hill_power: tuple[float, float]


@dataclass(frozen=True)
class CalibrationResult:
    parameters: ODEParameters
    residual_norm: float
    rank_correlation: float
    evaluations: int
    converged: bool
    message: str


class ParameterCodec:
    def __init__(self, template: ODEParameters) -> None:
        self.template = template
        self.rate_shape = template.production.shape
        self.matrix_shape = template.interaction.shape
        self.rate_size = template.production.numel()
        self.matrix_size = template.interaction.numel()

    def encode(self, parameters: ODEParameters) -> np.ndarray:
        tensors = (
            parameters.production,
            parameters.decay,
            parameters.interaction,
            parameters.half_saturation,
            parameters.hill_power,
        )
        return np.concatenate([tensor.detach().cpu().numpy().reshape(-1) for tensor in tensors])

    def decode(self, vector: np.ndarray) -> ODEParameters:
        cursor = 0

        def take(size: int, shape: torch.Size) -> Tensor:
            nonlocal cursor
            value = torch.from_numpy(vector[cursor : cursor + size].reshape(tuple(shape))).float()
            cursor += size
            return value

        production = take(self.rate_size, self.template.production.shape)
        decay = take(self.rate_size, self.template.decay.shape)
        interaction = take(self.matrix_size, self.template.interaction.shape)
        half_saturation = take(self.matrix_size, self.template.half_saturation.shape)
        hill_power = take(self.matrix_size, self.template.hill_power.shape)
        return ODEParameters(
            production=production,
            decay=decay,
            interaction=interaction,
            half_saturation=half_saturation,
            hill_power=hill_power,
            intervention=self.template.intervention,
            severity_knots=self.template.severity_knots,
            homeostatic=self.template.homeostatic,
            pathological=self.template.pathological,
            saddle=self.template.saddle,
            safety_ceiling=self.template.safety_ceiling,
            daily_dose_limit=self.template.daily_dose_limit,
        )

    def bounds(self, bounds: CalibrationBounds) -> tuple[np.ndarray, np.ndarray]:
        lower = np.concatenate(
            (
                np.full(self.rate_size, bounds.production[0]),
                np.full(self.rate_size, bounds.decay[0]),
                np.full(self.matrix_size, bounds.interaction[0]),
                np.full(self.matrix_size, bounds.half_saturation[0]),
                np.full(self.matrix_size, bounds.hill_power[0]),
            )
        )
        upper = np.concatenate(
            (
                np.full(self.rate_size, bounds.production[1]),
                np.full(self.rate_size, bounds.decay[1]),
                np.full(self.matrix_size, bounds.interaction[1]),
                np.full(self.matrix_size, bounds.half_saturation[1]),
                np.full(self.matrix_size, bounds.hill_power[1]),
            )
        )
        return lower, upper


class TrajectorySolver:
    def __init__(self, step_hours: float = 0.1) -> None:
        self.step_hours = step_hours

    def solve(self, parameters: ODEParameters, series: CalibrationSeries) -> np.ndarray:
        network = CytokineNetwork(parameters)
        state = torch.from_numpy(series.observations[0]).float().unsqueeze(0)
        action = torch.zeros((1, parameters.actions))
        severity = torch.tensor([series.severity])
        outputs = [state.squeeze(0).numpy()]
        current_time = float(series.times[0])
        for target_time in series.times[1:]:
            remaining = float(target_time) - current_time
            while remaining > 1e-9:
                increment = min(self.step_hours, remaining)
                integrator = RK4Integrator(network, increment, 1)
                with torch.no_grad():
                    state = integrator(state, action, severity)
                remaining -= increment
                current_time += increment
            outputs.append(state.squeeze(0).numpy())
        return np.stack(outputs)


class Calibrator:
    def __init__(
        self,
        template: ODEParameters,
        bounds: CalibrationBounds,
        step_hours: float = 0.1,
        loss_scale: float = 1.0,
    ) -> None:
        self.template = template
        self.bounds = bounds
        self.codec = ParameterCodec(template)
        self.solver = TrajectorySolver(step_hours)
        self.loss_scale = loss_scale

    def residuals(self, vector: np.ndarray, series: Sequence[CalibrationSeries]) -> np.ndarray:
        parameters = self.codec.decode(vector)
        residual_blocks = []
        for item in series:
            prediction = self.solver.solve(parameters, item)
            residual = (prediction - item.observations)[item.mask]
            residual_blocks.append(residual * self.loss_scale)
        return np.concatenate(residual_blocks)

    def fit(
        self,
        series: Sequence[CalibrationSeries],
        max_evaluations: int = 1000,
        callback: Callable[[np.ndarray], None] | None = None,
    ) -> CalibrationResult:
        for item in series:
            item.validate(self.template.cytokines)
        initial = self.codec.encode(self.template)
        limits = self.codec.bounds(self.bounds)
        result = least_squares(
            self.residuals,
            initial,
            bounds=limits,
            args=(series,),
            max_nfev=max_evaluations,
            method="trf",
            verbose=0,
        )
        if callback is not None:
            callback(result.x)
        parameters = self.codec.decode(result.x)
        observed_values = []
        predicted_values = []
        for item in series:
            prediction = self.solver.solve(parameters, item)
            observed_values.extend(item.observations[item.mask])
            predicted_values.extend(prediction[item.mask])
        correlation = spearmanr(observed_values, predicted_values).statistic
        return CalibrationResult(
            parameters=parameters,
            residual_norm=float(np.linalg.norm(result.fun)),
            rank_correlation=float(correlation),
            evaluations=int(result.nfev),
            converged=bool(result.success),
            message=str(result.message),
        )


def estimate_equilibria(
    parameters: ODEParameters,
    severity: float,
    starts: Tensor,
    iterations: int = 20000,
    step_hours: float = 0.1,
    tolerance: float = 1e-7,
) -> Tensor:
    network = CytokineNetwork(parameters)
    integrator = RK4Integrator(network, step_hours, 1)
    state = starts.clone()
    action = torch.zeros((state.shape[0], parameters.actions), dtype=state.dtype)
    severity_tensor = torch.full((state.shape[0],), severity, dtype=state.dtype)
    for _ in range(iterations):
        next_state = integrator(state, action, severity_tensor)
        if torch.max(torch.abs(next_state - state)).item() < tolerance:
            state = next_state
            break
        state = next_state
    rounded = torch.round(state / tolerance) * tolerance
    return torch.unique(rounded, dim=0)
