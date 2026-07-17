from __future__ import annotations

import argparse
import logging
from pathlib import Path

from trusthrl_sci.assembly import assemble_training
from trusthrl_sci.dynamics.network import ODEParameters
from trusthrl_sci.runtime import configure_logging, resolve_device, set_seed
from trusthrl_sci.specification import load_spec

LOGGER = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="trusthrl-train")
    value.add_argument("--config", type=Path, default=Path("presets/main.yaml"))
    value.add_argument("--parameters", type=Path)
    value.add_argument("--environments", type=int, default=16)
    value.add_argument("--output", type=Path, default=Path("artifacts/main"))
    return value


def main() -> None:
    arguments = parser().parse_args()
    configure_logging()
    specification = load_spec(arguments.config)
    parameter_path = arguments.parameters or Path(specification.parameter_file)
    parameters = ODEParameters.load(parameter_path)
    set_seed(specification.seed)
    device = resolve_device(specification.device)
    assembly = assemble_training(specification, parameters, device, arguments.environments)

    def report(progress: object) -> None:
        LOGGER.info("progress=%s", progress)

    progress = assembly.trainer.train(
        callback=report,
        checkpoint_interval=50000,
        checkpoint_root=arguments.output / "states",
    )
    assembly.trainer.save(arguments.output / "final.pt")
    LOGGER.info("completed=%s", progress)


if __name__ == "__main__":
    main()
