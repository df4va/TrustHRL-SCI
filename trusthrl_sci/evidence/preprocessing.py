from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass(frozen=True)
class FittedTransform:
    center: np.ndarray
    scale: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    feature_names: tuple[str, ...]


class CytokinePreprocessor:
    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = tuple(feature_names)
        self.minmax = MinMaxScaler(feature_range=(0, 1))
        self.standard = StandardScaler()
        self._fitted = False

    def validate(self, values: np.ndarray) -> None:
        if values.ndim != 2:
            raise ValueError("cytokine matrix must be two-dimensional")
        if values.shape[1] != len(self.feature_names):
            raise ValueError("cytokine feature count mismatch")
        if not np.all(np.isfinite(values)):
            raise ValueError("cytokine matrix contains non-finite values")
        if np.any(values < 0):
            raise ValueError("cytokine concentrations must be nonnegative")

    def fit(self, train_values: np.ndarray) -> CytokinePreprocessor:
        self.validate(train_values)
        scaled = self.minmax.fit_transform(train_values)
        self.standard.fit(scaled)
        self._fitted = True
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("preprocessor has not been fitted")
        self.validate(values)
        return self.standard.transform(self.minmax.transform(values))

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("preprocessor has not been fitted")
        return self.minmax.inverse_transform(self.standard.inverse_transform(values))

    def state(self) -> FittedTransform:
        if not self._fitted:
            raise RuntimeError("preprocessor has not been fitted")
        return FittedTransform(
            center=self.standard.mean_.copy(),
            scale=self.standard.scale_.copy(),
            minimum=self.minmax.data_min_.copy(),
            maximum=self.minmax.data_max_.copy(),
            feature_names=self.feature_names,
        )


class TranscriptomicPreprocessor:
    def __init__(self, pseudocount: float = 1.0) -> None:
        if pseudocount <= 0:
            raise ValueError("pseudocount must be positive")
        self.pseudocount = pseudocount
        self.control_mean: pd.Series | None = None

    def fit(
        self, expression: pd.DataFrame, control_samples: Sequence[str]
    ) -> TranscriptomicPreprocessor:
        missing = set(control_samples) - set(expression.columns)
        if missing:
            raise ValueError(f"missing control samples: {sorted(missing)}")
        logged = np.log2(expression.astype(float) + self.pseudocount)
        self.control_mean = logged.loc[:, list(control_samples)].mean(axis=1)
        return self

    def transform(self, expression: pd.DataFrame) -> pd.DataFrame:
        if self.control_mean is None:
            raise RuntimeError("preprocessor has not been fitted")
        missing = set(self.control_mean.index) - set(expression.index)
        if missing:
            raise ValueError("expression matrix lacks fitted genes")
        aligned = expression.loc[self.control_mean.index].astype(float)
        logged = np.log2(aligned + self.pseudocount)
        return logged.subtract(self.control_mean, axis=0)


@dataclass(frozen=True)
class CytokineGeneMap:
    cytokine: str
    genes: tuple[str, ...]
    aggregation: str = "mean"


def aggregate_gene_modules(
    expression: pd.DataFrame,
    mappings: Iterable[CytokineGeneMap],
) -> pd.DataFrame:
    rows: dict[str, pd.Series] = {}
    upper_index = {name.upper(): name for name in expression.index.astype(str)}
    for mapping in mappings:
        available = [
            upper_index[gene.upper()] for gene in mapping.genes if gene.upper() in upper_index
        ]
        if not available:
            rows[mapping.cytokine] = pd.Series(np.nan, index=expression.columns)
        elif mapping.aggregation == "mean":
            rows[mapping.cytokine] = expression.loc[available].mean(axis=0)
        elif mapping.aggregation == "median":
            rows[mapping.cytokine] = expression.loc[available].median(axis=0)
        elif mapping.aggregation == "maximum":
            rows[mapping.cytokine] = expression.loc[available].max(axis=0)
        else:
            raise ValueError(f"unknown aggregation {mapping.aggregation}")
    return pd.DataFrame(rows).transpose()


def fit_only_train(
    values: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    feature_names: Sequence[str],
) -> Mapping[str, np.ndarray]:
    processor = CytokinePreprocessor(feature_names).fit(values[train_indices])
    return {
        "train": processor.transform(values[train_indices]),
        "validation": processor.transform(values[validation_indices]),
        "test": processor.transform(values[test_indices]),
    }
