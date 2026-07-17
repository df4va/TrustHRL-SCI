from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricDatum:
    method: str
    dataset: str
    metric: str
    mean: float
    standard_deviation: float


@dataclass(frozen=True)
class AblationDatum:
    variant: str
    dataset: str
    hrs: float
    hrs_standard_deviation: float
    scr: float
    scr_standard_deviation: float


METHODS = (
    "Random",
    "Fixed Protocol",
    "PID",
    "MPC",
    "PPO",
    "SAC",
    "PPO-Lag",
    "CPO",
    "HIRO",
    "HAC",
    "TrustHRL",
)


DATASETS = (
    "Hellenbrand",
    "GSE5296",
    "GSE151371",
)


HRS_MEANS = np.asarray(
    [
        [18.2, 15.6, 12.4],
        [34.5, 29.1, 25.3],
        [42.3, 37.8, 33.9],
        [55.8, 50.2, 46.7],
        [52.4, 47.3, 43.8],
        [54.9, 49.8, 45.2],
        [51.7, 46.5, 42.9],
        [53.2, 48.1, 44.5],
        [62.1, 56.4, 51.8],
        [60.3, 54.7, 52.3],
        [73.4, 67.8, 62.5],
    ]
)


HRS_STANDARD_DEVIATIONS = np.asarray(
    [
        [2.8, 3.1, 2.5],
        [1.8, 2.3, 1.9],
        [2.1, 2.6, 2.4],
        [2.4, 2.9, 2.7],
        [3.2, 3.5, 3.1],
        [2.9, 3.3, 2.8],
        [2.7, 3.0, 2.6],
        [2.5, 2.8, 2.5],
        [2.6, 3.0, 2.7],
        [2.8, 3.2, 2.9],
        [2.1, 2.4, 2.6],
    ]
)


SCR_MEANS = np.asarray(
    [
        [42.1, 38.9, 35.7],
        [78.4, 74.6, 71.2],
        [82.7, 79.3, 76.1],
        [88.3, 85.1, 82.4],
        [71.6, 67.8, 64.2],
        [73.2, 69.4, 66.1],
        [93.8, 91.2, 89.5],
        [95.1, 93.4, 91.7],
        [76.3, 72.8, 69.5],
        [74.8, 71.2, 68.3],
        [97.2, 96.1, 94.8],
    ]
)


SCR_STANDARD_DEVIATIONS = np.asarray(
    [
        [5.9, 6.2, 5.4],
        [3.2, 3.8, 3.1],
        [2.8, 3.4, 2.9],
        [2.1, 2.6, 2.3],
        [4.7, 5.1, 4.4],
        [4.3, 4.8, 4.1],
        [1.9, 2.3, 2.1],
        [1.6, 1.9, 1.8],
        [3.8, 4.2, 3.6],
        [4.1, 4.5, 3.9],
        [1.1, 1.3, 1.5],
    ]
)


OVERSHOOT_MEANS = np.asarray(
    [
        [8.3, 6.7, 5.1],
        [22.7, 18.9, 15.6],
        [28.4, 24.1, 20.8],
        [36.9, 32.4, 29.1],
        [33.7, 29.2, 26.1],
        [35.2, 30.8, 27.5],
        [32.1, 27.9, 24.8],
        [34.5, 30.1, 27.0],
        [41.2, 36.3, 32.7],
        [39.8, 35.1, 33.5],
        [53.8, 48.2, 43.6],
    ]
)


OVERSHOOT_STANDARD_DEVIATIONS = np.asarray(
    [
        [3.2, 2.9, 2.3],
        [2.4, 2.7, 2.2],
        [2.7, 2.9, 2.5],
        [2.5, 2.8, 2.4],
        [3.1, 3.3, 2.8],
        [2.9, 3.1, 2.6],
        [2.6, 2.8, 2.4],
        [2.4, 2.7, 2.3],
        [2.8, 3.0, 2.6],
        [3.0, 3.2, 2.8],
        [2.3, 2.5, 2.4],
    ]
)


EFFICIENCY_MEANS = np.asarray(
    [
        [0.06, 0.05, 0.04],
        [0.11, 0.09, 0.08],
        [0.14, 0.12, 0.10],
        [0.19, 0.17, 0.15],
        [0.17, 0.15, 0.13],
        [0.18, 0.16, 0.14],
        [0.16, 0.14, 0.12],
        [0.17, 0.15, 0.13],
        [0.22, 0.19, 0.17],
        [0.21, 0.18, 0.17],
        [0.29, 0.26, 0.23],
    ]
)


EFFICIENCY_STANDARD_DEVIATIONS = np.asarray(
    [
        [0.02, 0.02, 0.01],
        [0.01, 0.01, 0.01],
        [0.02, 0.02, 0.01],
        [0.02, 0.02, 0.01],
        [0.02, 0.02, 0.02],
        [0.02, 0.02, 0.02],
        [0.02, 0.02, 0.01],
        [0.02, 0.02, 0.01],
        [0.02, 0.02, 0.02],
        [0.02, 0.02, 0.02],
        [0.02, 0.02, 0.02],
    ]
)


ABLATION_VARIANTS = (
    "Random Input",
    "Trivial PID",
    "Full Model",
    "No Hierarchy",
    "No CMDP Safety",
    "Four Variable ODE",
    "No Severity Conditioning",
    "Fixed Intervals",
    "Linear Kinetics",
    "Fixed Penalty",
    "Discount 0.95",
    "Two Phase",
    "Learning Rate 0.003",
    "Meta Frequency Doubled",
)


ABLATION_HRS = np.asarray(
    [
        [24.1, 20.8, 17.3],
        [42.3, 37.8, 33.9],
        [73.4, 67.8, 62.5],
        [54.2, 49.1, 44.7],
        [76.1, 70.3, 65.8],
        [66.8, 61.3, 56.9],
        [64.1, 58.7, 48.2],
        [67.2, 61.9, 57.4],
        [69.5, 64.1, 59.3],
        [72.8, 67.1, 61.7],
        [71.8, 66.3, 61.1],
        [70.1, 64.7, 59.8],
        [68.7, 63.2, 58.4],
        [72.6, 66.9, 61.8],
    ]
)


ABLATION_HRS_STANDARD_DEVIATIONS = np.asarray(
    [
        [3.5, 3.8, 3.2],
        [2.1, 2.6, 2.4],
        [2.1, 2.4, 2.6],
        [3.0, 3.3, 2.9],
        [2.3, 2.6, 2.8],
        [2.5, 2.8, 2.7],
        [2.8, 3.1, 3.4],
        [2.4, 2.7, 2.8],
        [2.3, 2.6, 2.7],
        [2.2, 2.5, 2.7],
        [2.3, 2.5, 2.7],
        [2.5, 2.8, 2.9],
        [3.4, 3.7, 3.5],
        [2.2, 2.5, 2.7],
    ]
)


ABLATION_SCR = np.asarray(
    [
        [51.3, 47.6, 44.2],
        [82.7, 79.3, 76.1],
        [97.2, 96.1, 94.8],
        [94.5, 92.8, 91.1],
        [68.4, 64.7, 61.3],
        [96.5, 95.2, 93.6],
        [95.8, 94.3, 87.6],
        [96.8, 95.5, 93.9],
        [96.9, 95.7, 94.1],
        [91.3, 89.7, 87.9],
        [96.9, 95.7, 94.2],
        [96.5, 95.1, 93.5],
        [95.6, 94.1, 92.7],
        [97.0, 95.8, 94.5],
    ]
)


ABLATION_SCR_STANDARD_DEVIATIONS = np.asarray(
    [
        [6.1, 6.5, 5.8],
        [2.8, 3.4, 2.9],
        [1.1, 1.3, 1.5],
        [1.8, 2.1, 2.0],
        [5.2, 5.6, 5.1],
        [1.2, 1.4, 1.6],
        [1.4, 1.6, 2.8],
        [1.2, 1.4, 1.6],
        [1.1, 1.3, 1.5],
        [2.4, 2.7, 2.6],
        [1.2, 1.3, 1.6],
        [1.3, 1.5, 1.7],
        [1.8, 2.0, 2.1],
        [1.1, 1.3, 1.5],
    ]
)


def primary_results() -> tuple[MetricDatum, ...]:
    matrices = {
        "HRS": (HRS_MEANS, HRS_STANDARD_DEVIATIONS),
        "SCR": (SCR_MEANS, SCR_STANDARD_DEVIATIONS),
        "overshoot_reduction": (OVERSHOOT_MEANS, OVERSHOOT_STANDARD_DEVIATIONS),
        "treatment_efficiency": (EFFICIENCY_MEANS, EFFICIENCY_STANDARD_DEVIATIONS),
    }
    records = []
    for metric, (means, deviations) in matrices.items():
        for method_index, method in enumerate(METHODS):
            for dataset_index, dataset in enumerate(DATASETS):
                records.append(
                    MetricDatum(
                        method,
                        dataset,
                        metric,
                        float(means[method_index, dataset_index]),
                        float(deviations[method_index, dataset_index]),
                    )
                )
    return tuple(records)


def ablation_results() -> tuple[AblationDatum, ...]:
    records = []
    for variant_index, variant in enumerate(ABLATION_VARIANTS):
        for dataset_index, dataset in enumerate(DATASETS):
            records.append(
                AblationDatum(
                    variant,
                    dataset,
                    float(ABLATION_HRS[variant_index, dataset_index]),
                    float(ABLATION_HRS_STANDARD_DEVIATIONS[variant_index, dataset_index]),
                    float(ABLATION_SCR[variant_index, dataset_index]),
                    float(ABLATION_SCR_STANDARD_DEVIATIONS[variant_index, dataset_index]),
                )
            )
    return tuple(records)


def index_results(records: Iterable[MetricDatum]) -> Mapping[tuple[str, str, str], MetricDatum]:
    return {(item.method, item.dataset, item.metric): item for item in records}


def compare_to_reference(
    observed: Mapping[tuple[str, str, str], float],
    tolerance_standard_deviations: float = 2.0,
) -> dict[tuple[str, str, str], bool]:
    reference = index_results(primary_results())
    result = {}
    for key, value in observed.items():
        datum = reference[key]
        tolerance = tolerance_standard_deviations * datum.standard_deviation
        result[key] = abs(value - datum.mean) <= tolerance
    return result
