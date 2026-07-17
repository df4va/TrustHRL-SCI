from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentSpec:
    seed: int
    seeds: tuple[int, ...]
    device: str
    precision: str
    cytokines: int
    actions: int
    horizon_hours: int
    rl_step_hours: float
    ode_step_hours: float
    meta_interval_hours: int
    hidden_width: int
    hidden_layers: int
    environment_steps: int
    rollout_size: int
    minibatch_size: int
    ppo_epochs: int
    learning_rate: float
    discount: float
    gae_lambda: float
    clip_ratio: float
    entropy_coefficient: float
    value_coefficient: float
    gradient_clip: float
    adam_beta1: float
    adam_beta2: float
    lagrange_learning_rate: float
    safety_budget: float
    immunodeficiency_floor: float
    phase_boundaries: tuple[int, ...]
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    normalization: str
    cytokine_scaling: str
    transcriptomic_transform: str
    artifact_root: str
    parameter_file: str
    hierarchy: bool = True
    capacity_match: bool = False
    safety_layer: bool = True
    severity_conditioning: bool = True
    phase_selection: str = "learned"
    kinetics: str = "hill"
    constraint_solver: str = "lagrangian"
    penalty_coefficient: float = 0.0
    selected_cytokines: tuple[str, ...] = ()
    grid_parameter: str = ""
    grid_values: tuple[float, ...] = ()
    selection_metric: str = ""
    method: str = "trusthrl"
    trust_region: float = 0.0
    phase_count: int = 3

    def validate(self) -> None:
        if self.cytokines <= 0:
            raise ValueError("cytokines must be positive")
        if self.actions <= 0:
            raise ValueError("actions must be positive")
        if self.horizon_hours <= 0:
            raise ValueError("horizon must be positive")
        if self.rl_step_hours <= 0 or self.ode_step_hours <= 0:
            raise ValueError("time increments must be positive")
        ratio = self.rl_step_hours / self.ode_step_hours
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("RL step must contain an integer number of ODE steps")
        if self.phase_boundaries[0] != 0:
            raise ValueError("phase boundaries must begin at zero")
        if self.phase_boundaries[-1] != self.horizon_hours:
            raise ValueError("phase boundaries must end at horizon")
        if len(self.phase_boundaries) != self.phase_count + 1:
            raise ValueError("phase count and boundaries must agree")
        if tuple(sorted(self.phase_boundaries)) != self.phase_boundaries:
            raise ValueError("phase boundaries must be ordered")
        fractions = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(fractions - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to one")
        if self.rollout_size % self.minibatch_size:
            raise ValueError("rollout size must be divisible by minibatch size")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must lie in (0, 1]")
        if not 0 < self.gae_lambda <= 1:
            raise ValueError("GAE lambda must lie in (0, 1]")
        if not 0 < self.safety_budget < 1:
            raise ValueError("safety budget must lie in (0, 1)")

    @property
    def state_width(self) -> int:
        return self.cytokines + int(self.severity_conditioning)

    @property
    def meta_state_width(self) -> int:
        return self.state_width + 2

    @property
    def ode_substeps(self) -> int:
        return round(self.rl_step_hours / self.ode_step_hours)

    @property
    def episode_steps(self) -> int:
        return round(self.horizon_hours / self.rl_step_hours)

    @property
    def episodes(self) -> int:
        return self.environment_steps // self.episode_steps


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"configuration at {path} must be a mapping")
    return {str(key): value for key, value in data.items()}


def _resolve(path: Path, seen: frozenset[Path]) -> dict[str, Any]:
    canonical = path.resolve()
    if canonical in seen:
        raise ValueError(f"configuration inheritance cycle at {canonical}")
    values = _read_yaml(canonical)
    parent = values.pop("inherit", None)
    if parent is None:
        return values
    if not isinstance(parent, str):
        raise ValueError("inherit must be a file name")
    merged = _resolve(canonical.parent / parent, seen | {canonical})
    merged.update(values)
    return merged


def _coerce(values: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(values)
    for key in ("seeds", "phase_boundaries", "selected_cytokines", "grid_values"):
        if key in result:
            result[key] = tuple(result[key])
    return result


def load_spec(path: str | Path) -> ExperimentSpec:
    values = _coerce(_resolve(Path(path), frozenset()))
    allowed = {field.name for field in fields(ExperimentSpec)}
    unexpected = set(values) - allowed
    if unexpected:
        raise ValueError(f"unknown configuration fields: {sorted(unexpected)}")
    spec = ExperimentSpec(**values)
    spec.validate()
    return spec


def with_overrides(spec: ExperimentSpec, overrides: Mapping[str, Any]) -> ExperimentSpec:
    values = {field.name: getattr(spec, field.name) for field in fields(spec)}
    unknown = set(overrides) - set(values)
    if unknown:
        raise ValueError(f"unknown overrides: {sorted(unknown)}")
    values.update(overrides)
    updated = ExperimentSpec(**values)
    updated.validate()
    return updated
