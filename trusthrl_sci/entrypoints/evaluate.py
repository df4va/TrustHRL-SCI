from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from trusthrl_sci.assessment.metrics import compute_episode_metrics
from trusthrl_sci.dynamics.catalog import pro_inflammatory_indices


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="trusthrl-evaluate")
    value.add_argument("--trajectories", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--reward-threshold", type=float, default=0.8)
    return value


def main() -> None:
    arguments = parser().parse_args()
    with np.load(arguments.trajectories, allow_pickle=False) as data:
        metrics = compute_episode_metrics(
            controlled_states=data["controlled_states"],
            uncontrolled_states=data["uncontrolled_states"],
            homeostatic=data["homeostatic"],
            initial=data["initial"],
            actions=data["actions"],
            costs=data["costs"],
            pro_inflammatory_indices=np.asarray(pro_inflammatory_indices()),
            reward_threshold=arguments.reward_threshold,
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
