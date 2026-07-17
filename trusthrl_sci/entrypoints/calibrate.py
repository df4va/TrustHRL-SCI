from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from trusthrl_sci.dynamics.calibration import (
    CalibrationBounds,
    CalibrationSeries,
    Calibrator,
)
from trusthrl_sci.dynamics.network import ODEParameters
from trusthrl_sci.runtime import configure_logging

LOGGER = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="trusthrl-calibrate")
    value.add_argument("--template", type=Path, required=True)
    value.add_argument("--series", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--max-evaluations", type=int, default=1000)
    value.add_argument("--step-hours", type=float, default=0.1)
    return value


def read_series(path: Path, cytokines: int) -> tuple[CalibrationSeries, ...]:
    with np.load(path, allow_pickle=False) as data:
        severities = data["severities"]
        times = data["times"]
        observations = data["observations"]
        masks = data["masks"]
    if observations.ndim != 3 or observations.shape[-1] != cytokines:
        raise ValueError("calibration observations have incorrect shape")
    result = []
    for index, severity in enumerate(severities):
        result.append(
            CalibrationSeries(
                severity=float(severity),
                times=times[index],
                observations=observations[index],
                mask=masks[index].astype(bool),
            )
        )
    return tuple(result)


def main() -> None:
    arguments = parser().parse_args()
    configure_logging()
    template = ODEParameters.load(arguments.template)
    series = read_series(arguments.series, template.cytokines)
    bounds = CalibrationBounds(
        production=(1e-8, 100.0),
        decay=(1e-8, 100.0),
        interaction=(-10.0, 10.0),
        half_saturation=(1e-8, 1000.0),
        hill_power=(0.1, 8.0),
    )
    calibrator = Calibrator(template, bounds, arguments.step_hours)
    result = calibrator.fit(series, arguments.max_evaluations)
    result.parameters.save(arguments.output)
    report = {
        "residual_norm": result.residual_norm,
        "rank_correlation": result.rank_correlation,
        "evaluations": result.evaluations,
        "converged": result.converged,
        "message": result.message,
    }
    report_path = arguments.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("calibration=%s", report)


if __name__ == "__main__":
    main()
