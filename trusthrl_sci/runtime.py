from __future__ import annotations

import json
import logging
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


@dataclass(frozen=True)
class RunIdentity:
    name: str
    seed: int
    started_at: str
    configuration_digest: str


@dataclass(frozen=True)
class SavedState:
    step: int
    episode: int
    seed: int
    model: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    lagrange: Mapping[str, Any]
    scheduler: Mapping[str, Any]


def atomic_torch_save(payload: Mapping[str, Any], destination: str | Path) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=target.parent, suffix=".partial")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_save(payload: Mapping[str, Any], destination: str | Path) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=target.parent, suffix=".partial")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def dataclass_payload(value: object) -> dict[str, Any]:
    return dict(asdict(value))


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    if "seed" not in payload:
        raise ValueError("checkpoint lacks seed")
    set_seed(int(payload["seed"]))
    return payload
