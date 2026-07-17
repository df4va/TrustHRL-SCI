from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from trusthrl_sci.assessment.metrics import EpisodeMetrics, summarize_seeds
from trusthrl_sci.assessment.statistics import Comparison, wilcoxon_comparisons


@dataclass(frozen=True)
class MethodRun:
    method: str
    dataset: str
    seed: int
    metrics: EpisodeMetrics
    split_identifier: str


@dataclass(frozen=True)
class MethodSummary:
    method: str
    dataset: str
    seeds: tuple[int, ...]
    values: Mapping[str, tuple[float, float]]


@dataclass(frozen=True)
class PrimaryComparison:
    comparator: str
    dataset: str
    metric: str
    result: Comparison


class ResultLedger:
    def __init__(self) -> None:
        self.runs: list[MethodRun] = []

    def append(self, run: MethodRun) -> None:
        duplicate = any(
            item.method == run.method
            and item.dataset == run.dataset
            and item.seed == run.seed
            and item.split_identifier == run.split_identifier
            for item in self.runs
        )
        if duplicate:
            raise ValueError("duplicate method run")
        self.runs.append(run)

    def select(self, method: str, dataset: str) -> tuple[MethodRun, ...]:
        return tuple(
            item for item in self.runs if item.method == method and item.dataset == dataset
        )

    def validate_alignment(self, methods: Sequence[str], dataset: str) -> None:
        reference: set[tuple[int, str]] | None = None
        for method in methods:
            keys = {(item.seed, item.split_identifier) for item in self.select(method, dataset)}
            if reference is None:
                reference = keys
            elif keys != reference:
                raise ValueError(f"run alignment differs for {method} on {dataset}")

    def summary(self, method: str, dataset: str) -> MethodSummary:
        runs = self.select(method, dataset)
        if not runs:
            raise ValueError("no runs selected")
        metrics = [run.metrics for run in runs]
        seeds = tuple(run.seed for run in runs)
        return MethodSummary(method, dataset, seeds, summarize_seeds(metrics))

    def metric_vector(self, method: str, dataset: str, metric: str) -> np.ndarray:
        runs = sorted(self.select(method, dataset), key=lambda item: item.seed)
        if not runs:
            raise ValueError("no runs selected")
        if metric not in EpisodeMetrics.__dataclass_fields__:
            raise ValueError(f"unknown metric {metric}")
        return np.asarray([getattr(run.metrics, metric) for run in runs])

    def primary_comparisons(
        self,
        reference: str,
        comparators: Sequence[str],
        datasets: Sequence[str],
        metric: str = "homeostasis_restoration_score",
    ) -> tuple[PrimaryComparison, ...]:
        reference_vectors = []
        candidate_vectors = []
        labels = []
        for dataset in datasets:
            self.validate_alignment((reference, *comparators), dataset)
            reference_vector = self.metric_vector(reference, dataset, metric)
            for comparator in comparators:
                reference_vectors.append(reference_vector)
                candidate_vectors.append(self.metric_vector(comparator, dataset, metric))
                labels.append((comparator, dataset))
        raw_results = []
        p_values = []
        for reference_vector, candidate_vector in zip(
            reference_vectors, candidate_vectors, strict=True
        ):
            result = wilcoxon_comparisons(reference_vector, (candidate_vector,))[0]
            raw_results.append(result)
            p_values.append(result.uncorrected_p)
        from trusthrl_sci.assessment.statistics import holm_bonferroni

        corrected = holm_bonferroni(p_values)
        results = []
        for label, raw, adjusted in zip(labels, raw_results, corrected, strict=True):
            comparator, dataset = label
            result = Comparison(
                statistic=raw.statistic,
                uncorrected_p=raw.uncorrected_p,
                corrected_p=float(adjusted),
                effect_size=raw.effect_size,
                significant=bool(adjusted < 0.05),
            )
            results.append(PrimaryComparison(comparator, dataset, metric, result))
        return tuple(results)


def cross_validation(
    folds: Iterable[tuple[np.ndarray, np.ndarray]],
    train: Callable[[np.ndarray], object],
    evaluate: Callable[[object, np.ndarray], float],
) -> np.ndarray:
    scores = []
    for train_indices, test_indices in folds:
        model = train(train_indices)
        scores.append(evaluate(model, test_indices))
    return np.asarray(scores)


def tuning_grid(
    learning_rates: Sequence[float] = (1e-4, 3e-4, 1e-3),
    discounts: Sequence[float] = (0.95, 0.99),
    seeds: Sequence[int] = (17, 29, 43),
) -> tuple[dict[str, float | int], ...]:
    return tuple(
        {
            "learning_rate": learning_rate,
            "discount": discount,
            "seed": seed,
        }
        for learning_rate in learning_rates
        for discount in discounts
        for seed in seeds
    )


def select_validation_best(
    configurations: Sequence[Mapping[str, float | int]],
    validation_scores: Sequence[float],
) -> Mapping[str, float | int]:
    if len(configurations) != len(validation_scores):
        raise ValueError("configurations and scores must align")
    if not configurations:
        raise ValueError("configurations cannot be empty")
    index = int(np.argmax(validation_scores))
    return configurations[index]


def verify_test_isolation(
    tuning_indices: Sequence[int],
    test_indices: Sequence[int],
) -> None:
    overlap = np.intersect1d(tuning_indices, test_indices)
    if overlap.size:
        raise ValueError("test examples entered hyperparameter selection")


def phase_slice(values: np.ndarray, phase: int) -> np.ndarray:
    boundaries = (0, 6, 72, 168)
    if phase not in (0, 1, 2):
        raise ValueError("phase must be zero, one, or two")
    return values[boundaries[phase] : boundaries[phase + 1]]


def worst_episode_safety(costs: np.ndarray) -> float:
    if costs.ndim != 3 or costs.shape[-1] != 3:
        raise ValueError("cost array must have episode, time, constraint axes")
    rates = 100 * (1 - costs.mean(axis=(1, 2)))
    return float(rates.min())


def per_constraint_violation(costs: np.ndarray) -> np.ndarray:
    if costs.shape[-1] != 3:
        raise ValueError("three constraint channels are required")
    axes = tuple(range(costs.ndim - 1))
    return costs.mean(axis=axes)
