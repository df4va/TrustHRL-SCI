from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np


@dataclass(frozen=True)
class EpisodeMetrics:
    homeostasis_restoration_score: float
    safety_constraint_rate: float
    overshoot_reduction: float
    treatment_efficiency: float
    concentration_violation_rate: float
    dose_violation_rate: float
    immunodeficiency_violation_rate: float


def normalized_reward(
    states: np.ndarray, homeostatic: np.ndarray, initial: np.ndarray
) -> np.ndarray:
    numerator = np.linalg.norm(states - homeostatic, axis=-1)
    denominator = np.linalg.norm(initial - homeostatic)
    return 1 - numerator / max(denominator, np.finfo(float).eps)


def homeostasis_restoration_score(rewards: np.ndarray, threshold: float) -> float:
    if rewards.size == 0:
        raise ValueError("rewards cannot be empty")
    return float(100 * np.mean(rewards > threshold))


def safety_constraint_rate(costs: np.ndarray) -> float:
    if costs.ndim != 2 or costs.shape[1] != 3:
        raise ValueError("cost matrix must have three columns")
    return float(100 * (1 - costs.mean()))


def overshoot(
    states: np.ndarray,
    baseline: np.ndarray,
    pro_inflammatory_indices: np.ndarray,
) -> float:
    selected = states[:, pro_inflammatory_indices]
    reference = baseline[:, pro_inflammatory_indices]
    excess = np.maximum(selected - reference, 0)
    return float(np.trapz(excess.sum(axis=-1)))


def overshoot_reduction(
    controlled: np.ndarray,
    uncontrolled: np.ndarray,
    baseline: np.ndarray,
    pro_inflammatory_indices: np.ndarray,
) -> float:
    uncontrolled_area = overshoot(uncontrolled, baseline, pro_inflammatory_indices)
    controlled_area = overshoot(controlled, baseline, pro_inflammatory_indices)
    return float(100 * (uncontrolled_area - controlled_area) / max(uncontrolled_area, 1e-12))


def treatment_efficiency(rewards: np.ndarray, actions: np.ndarray) -> float:
    dose = float(actions.sum())
    return float(rewards.sum() / max(dose, 1e-12))


def compute_episode_metrics(
    controlled_states: np.ndarray,
    uncontrolled_states: np.ndarray,
    homeostatic: np.ndarray,
    initial: np.ndarray,
    actions: np.ndarray,
    costs: np.ndarray,
    pro_inflammatory_indices: np.ndarray,
    reward_threshold: float,
) -> EpisodeMetrics:
    rewards = normalized_reward(controlled_states, homeostatic, initial)
    baseline = np.broadcast_to(homeostatic, controlled_states.shape)
    return EpisodeMetrics(
        homeostasis_restoration_score=homeostasis_restoration_score(rewards, reward_threshold),
        safety_constraint_rate=safety_constraint_rate(costs),
        overshoot_reduction=overshoot_reduction(
            controlled_states,
            uncontrolled_states,
            baseline,
            pro_inflammatory_indices,
        ),
        treatment_efficiency=treatment_efficiency(rewards, actions),
        concentration_violation_rate=float(costs[:, 0].mean()),
        dose_violation_rate=float(costs[:, 1].mean()),
        immunodeficiency_violation_rate=float(costs[:, 2].mean()),
    )


@dataclass(frozen=True)
class PhaseMetrics:
    acute: EpisodeMetrics
    subacute: EpisodeMetrics
    chronic: EpisodeMetrics


def phase_metrics(
    controlled_states: np.ndarray,
    uncontrolled_states: np.ndarray,
    homeostatic: np.ndarray,
    initial: np.ndarray,
    actions: np.ndarray,
    costs: np.ndarray,
    pro_inflammatory_indices: np.ndarray,
    reward_threshold: float,
    boundaries: tuple[int, int, int, int] = (0, 6, 72, 168),
) -> PhaseMetrics:
    values = []
    for start, end in pairwise(boundaries):
        values.append(
            compute_episode_metrics(
                controlled_states[start:end],
                uncontrolled_states[start:end],
                homeostatic,
                initial,
                actions[start:end],
                costs[start:end],
                pro_inflammatory_indices,
                reward_threshold,
            )
        )
    return PhaseMetrics(*values)


def summarize_seeds(metrics: list[EpisodeMetrics]) -> dict[str, tuple[float, float]]:
    if not metrics:
        raise ValueError("metrics cannot be empty")
    result = {}
    for field_name in EpisodeMetrics.__dataclass_fields__:
        values = np.asarray([getattr(item, field_name) for item in metrics])
        result[field_name] = (float(values.mean()), float(values.std(ddof=1)))
    return result
