from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor

from trusthrl_sci.dynamics.network import ODEParameters


@dataclass(frozen=True)
class Perturbation:
    group: str
    fraction: float
    parameter_count: int


@dataclass(frozen=True)
class SensitivityOutcome:
    perturbation: Perturbation
    baseline_hrs: float
    perturbed_hrs: float
    hrs_delta: float
    baseline_concordance: float
    perturbed_concordance: float
    concordance_delta: float


def scale_selected(values: Tensor, indices: Tensor, fraction: float) -> Tensor:
    result = values.clone()
    result[indices] *= 1 + fraction
    return result


def perturb_production(
    parameters: ODEParameters,
    cytokine_indices: Iterable[int],
    fraction: float,
) -> ODEParameters:
    production = parameters.production.clone()
    indices = torch.tensor(tuple(cytokine_indices), dtype=torch.long)
    production[:, indices] *= 1 + fraction
    return replace(parameters, production=production)


def perturb_decay(
    parameters: ODEParameters,
    cytokine_indices: Iterable[int],
    fraction: float,
) -> ODEParameters:
    decay = parameters.decay.clone()
    indices = torch.tensor(tuple(cytokine_indices), dtype=torch.long)
    decay[:, indices] *= 1 + fraction
    return replace(parameters, decay=decay)


def perturb_interactions(
    parameters: ODEParameters,
    edges: Iterable[tuple[int, int]],
    fraction: float,
) -> ODEParameters:
    interaction = parameters.interaction.clone()
    for target, source in edges:
        interaction[:, target, source] *= 1 + fraction
    return replace(parameters, interaction=interaction)


def perturb_hill_power(
    parameters: ODEParameters,
    edges: Iterable[tuple[int, int]],
    fraction: float,
) -> ODEParameters:
    hill_power = parameters.hill_power.clone()
    for target, source in edges:
        hill_power[:, target, source] *= 1 + fraction
    return replace(parameters, hill_power=hill_power)


def perturb_intervention(
    parameters: ODEParameters,
    actions: Iterable[int],
    fraction: float,
) -> ODEParameters:
    intervention = parameters.intervention.clone()
    indices = torch.tensor(tuple(actions), dtype=torch.long)
    intervention[:, :, indices] *= 1 + fraction
    return replace(parameters, intervention=intervention)


def perturb_homeostasis(parameters: ODEParameters, fraction: float) -> ODEParameters:
    return replace(parameters, homeostatic=parameters.homeostatic * (1 + fraction))


def perturb_severity_scaling(parameters: ODEParameters, fraction: float) -> ODEParameters:
    midpoint = parameters.production.mean(dim=0, keepdim=True)
    production = midpoint + (1 + fraction) * (parameters.production - midpoint)
    decay_midpoint = parameters.decay.mean(dim=0, keepdim=True)
    decay = decay_midpoint + (1 + fraction) * (parameters.decay - decay_midpoint)
    return replace(parameters, production=production, decay=decay)


def relative_change(baseline: float, perturbed: float) -> float:
    return 100 * (perturbed - baseline) / max(abs(baseline), 1e-12)


class SensitivityStudy:
    def __init__(
        self,
        evaluator: Callable[[ODEParameters], Mapping[str, float]],
        baseline: ODEParameters,
    ) -> None:
        self.evaluator = evaluator
        self.baseline = baseline

    def evaluate(
        self,
        group: str,
        parameter_count: int,
        transformer: Callable[[ODEParameters, float], ODEParameters],
        fractions: Iterable[float] = (-0.2, 0.2),
    ) -> tuple[SensitivityOutcome, ...]:
        baseline_metrics = self.evaluator(self.baseline)
        outcomes = []
        for fraction in fractions:
            modified = transformer(self.baseline, fraction)
            modified.validate()
            metrics = self.evaluator(modified)
            perturbation = Perturbation(group, fraction, parameter_count)
            outcomes.append(
                SensitivityOutcome(
                    perturbation=perturbation,
                    baseline_hrs=baseline_metrics["hrs"],
                    perturbed_hrs=metrics["hrs"],
                    hrs_delta=metrics["hrs"] - baseline_metrics["hrs"],
                    baseline_concordance=baseline_metrics["concordance"],
                    perturbed_concordance=metrics["concordance"],
                    concordance_delta=(metrics["concordance"] - baseline_metrics["concordance"]),
                )
            )
        return tuple(outcomes)


def classify_sensitivity(outcomes: Iterable[SensitivityOutcome], threshold: float = 2.0) -> str:
    maximum = max(abs(item.hrs_delta) for item in outcomes)
    return "sensitive" if maximum >= threshold else "stable"


def interaction_ratio(combined_delta: float, first_delta: float, second_delta: float) -> float:
    denominator = abs(first_delta) + abs(second_delta)
    return abs(combined_delta) / max(denominator, 1e-12)


def saturation_gap(full_value: float, held_out_value: float) -> float:
    return full_value - held_out_value


def degradation_classification(in_distribution: float, out_of_distribution: float) -> str:
    degradation = in_distribution - out_of_distribution
    if degradation <= 5:
        return "graceful"
    if degradation <= 10:
        return "moderate"
    if degradation <= 20:
        return "severe"
    return "critical"


def rank_parameter_groups(outcomes: Iterable[SensitivityOutcome]) -> tuple[str, ...]:
    grouped: dict[str, list[float]] = {}
    for item in outcomes:
        grouped.setdefault(item.perturbation.group, []).append(abs(item.hrs_delta))
    scores = {group: float(np.max(values)) for group, values in grouped.items()}
    return tuple(sorted(scores, key=scores.get, reverse=True))
