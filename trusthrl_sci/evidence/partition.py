from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold


@dataclass(frozen=True)
class Partition:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def validate(self, size: int) -> None:
        combined = np.concatenate((self.train, self.validation, self.test))
        if np.any(combined < 0) or np.any(combined >= size):
            raise ValueError("partition index out of range")
        if np.unique(combined).size != combined.size:
            raise ValueError("partitions overlap")
        if combined.size != size:
            raise ValueError("partitions do not cover all examples")


def random_partition(
    size: int,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.15,
    seed: int = 17,
) -> Partition:
    if size <= 0:
        raise ValueError("size must be positive")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("test fraction must be positive")
    generator = np.random.default_rng(seed)
    order = generator.permutation(size)
    train_end = round(size * train_fraction)
    validation_end = train_end + round(size * validation_fraction)
    partition = Partition(
        order[:train_end], order[train_end:validation_end], order[validation_end:]
    )
    partition.validate(size)
    return partition


def stratified_partition(
    labels: Sequence[int],
    train_fraction: float = 0.60,
    validation_fraction: float = 0.15,
    seed: int = 17,
) -> Partition:
    label_array = np.asarray(labels)
    generator = np.random.default_rng(seed)
    train_parts = []
    validation_parts = []
    test_parts = []
    for label in np.unique(label_array):
        indices = np.flatnonzero(label_array == label)
        indices = generator.permutation(indices)
        train_end = round(indices.size * train_fraction)
        validation_end = train_end + round(indices.size * validation_fraction)
        train_parts.append(indices[:train_end])
        validation_parts.append(indices[train_end:validation_end])
        test_parts.append(indices[validation_end:])
    partition = Partition(
        generator.permutation(np.concatenate(train_parts)),
        generator.permutation(np.concatenate(validation_parts)),
        generator.permutation(np.concatenate(test_parts)),
    )
    partition.validate(label_array.size)
    return partition


def five_fold_indices(
    size: int,
    seed: int = 17,
    labels: Sequence[int] | None = None,
    groups: Sequence[int] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(size)
    if groups is not None:
        group_array = np.asarray(groups)
        splitter = GroupKFold(n_splits=5)
        yield from splitter.split(indices, groups=group_array)
    elif labels is not None:
        label_array = np.asarray(labels)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        yield from splitter.split(indices, label_array)
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
        yield from splitter.split(indices)


def nested_validation(
    train: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < fraction < 1:
        raise ValueError("fraction must lie in (0, 1)")
    generator = np.random.default_rng(seed)
    order = generator.permutation(train)
    validation_size = max(1, round(order.size * fraction))
    return order[validation_size:], order[:validation_size]


def grouped_time_partition(
    subjects: Sequence[str],
    times: Sequence[float],
    seed: int = 17,
) -> Partition:
    subjects_array = np.asarray(subjects)
    times_array = np.asarray(times)
    if subjects_array.shape != times_array.shape:
        raise ValueError("subjects and times must align")
    unique_subjects = np.unique(subjects_array)
    subject_partition = random_partition(unique_subjects.size, seed=seed)
    train_subjects = unique_subjects[subject_partition.train]
    validation_subjects = unique_subjects[subject_partition.validation]
    test_subjects = unique_subjects[subject_partition.test]
    partition = Partition(
        np.flatnonzero(np.isin(subjects_array, train_subjects)),
        np.flatnonzero(np.isin(subjects_array, validation_subjects)),
        np.flatnonzero(np.isin(subjects_array, test_subjects)),
    )
    partition.validate(subjects_array.size)
    return partition
