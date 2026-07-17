from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int


@dataclass(frozen=True)
class Comparison:
    statistic: float
    uncorrected_p: float
    corrected_p: float
    effect_size: float
    significant: bool


def bootstrap_interval(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 17,
) -> ConfidenceInterval:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a nonempty vector")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    generator = np.random.default_rng(seed)
    samples = generator.choice(array, size=(resamples, array.size), replace=True)
    estimates = np.asarray([statistic(sample) for sample in samples])
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(estimates, [alpha, 1 - alpha])
    return ConfidenceInterval(
        estimate=float(statistic(array)),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        resamples=resamples,
    )


def paired_bootstrap_difference(
    first: Sequence[float],
    second: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 17,
) -> ConfidenceInterval:
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    if first_array.shape != second_array.shape:
        raise ValueError("paired values must align")
    return bootstrap_interval(
        first_array - second_array,
        np.mean,
        confidence,
        resamples,
        seed,
    )


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values)
    corrected = np.empty_like(values)
    running = 0.0
    count = values.size
    for rank, index in enumerate(order):
        adjusted = min(1.0, (count - rank) * values[index])
        running = max(running, adjusted)
        corrected[index] = running
    return corrected


def cohens_d(first: Sequence[float], second: Sequence[float], paired: bool = True) -> float:
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    if first_array.shape != second_array.shape and paired:
        raise ValueError("paired samples must align")
    if paired:
        differences = first_array - second_array
        return float(differences.mean() / max(differences.std(ddof=1), 1e-12))
    numerator = first_array.mean() - second_array.mean()
    degrees = first_array.size + second_array.size - 2
    pooled_variance = (
        (first_array.size - 1) * first_array.var(ddof=1)
        + (second_array.size - 1) * second_array.var(ddof=1)
    ) / degrees
    return float(numerator / max(np.sqrt(pooled_variance), 1e-12))


def wilcoxon_comparisons(
    reference: Sequence[float],
    candidates: Sequence[Sequence[float]],
    alpha: float = 0.05,
) -> tuple[Comparison, ...]:
    reference_array = np.asarray(reference, dtype=float)
    raw = []
    statistics = []
    effects = []
    for candidate in candidates:
        candidate_array = np.asarray(candidate, dtype=float)
        if candidate_array.shape != reference_array.shape:
            raise ValueError("comparison samples must align")
        result = stats.wilcoxon(reference_array, candidate_array, alternative="two-sided")
        statistics.append(float(result.statistic))
        raw.append(float(result.pvalue))
        effects.append(cohens_d(reference_array, candidate_array, paired=True))
    corrected = holm_bonferroni(raw, alpha)
    return tuple(
        Comparison(statistic, p_value, adjusted, effect, bool(adjusted < alpha))
        for statistic, p_value, adjusted, effect in zip(
            statistics, raw, corrected, effects, strict=True
        )
    )


def weighted_kappa(first: Sequence[int], second: Sequence[int], categories: int = 4) -> float:
    first_array = np.asarray(first, dtype=int)
    second_array = np.asarray(second, dtype=int)
    if first_array.shape != second_array.shape:
        raise ValueError("ratings must align")
    confusion = np.zeros((categories, categories), dtype=float)
    np.add.at(confusion, (first_array, second_array), 1)
    confusion /= max(confusion.sum(), 1)
    first_marginal = confusion.sum(axis=1)
    second_marginal = confusion.sum(axis=0)
    expected = np.outer(first_marginal, second_marginal)
    indices = np.arange(categories)
    weights = np.square(indices[:, None] - indices[None, :]) / max((categories - 1) ** 2, 1)
    observed_disagreement = float((weights * confusion).sum())
    expected_disagreement = float((weights * expected).sum())
    return float(1 - observed_disagreement / max(expected_disagreement, 1e-12))


def mcnemar(first_correct: Sequence[bool], second_correct: Sequence[bool]) -> tuple[float, float]:
    first = np.asarray(first_correct, dtype=bool)
    second = np.asarray(second_correct, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("outcome vectors must align")
    first_only = int(np.sum(first & ~second))
    second_only = int(np.sum(~first & second))
    discordant = first_only + second_only
    if discordant == 0:
        return 0.0, 1.0
    statistic = (abs(first_only - second_only) - 1) ** 2 / discordant
    p_value = float(stats.chi2.sf(statistic, 1))
    return float(statistic), p_value


def concordance_grade(prediction: float, observation: float) -> int:
    denominator = max(abs(observation), 1e-12)
    relative_error = abs(prediction - observation) / denominator
    if relative_error <= 0.10:
        return 3
    if relative_error <= 0.25:
        return 2
    if np.sign(prediction) == np.sign(observation):
        return 1
    return 0


def graduated_concordance(grades: Sequence[int]) -> float:
    values = np.asarray(grades, dtype=float)
    if np.any(values < 0) or np.any(values > 3):
        raise ValueError("grades must lie between zero and three")
    return float(100 * values.mean() / 3)
