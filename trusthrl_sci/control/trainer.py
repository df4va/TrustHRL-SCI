from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from trusthrl_sci.contracts import Transition
from trusthrl_sci.control.environment import InflammationEnvironment, VectorEpisodeLedger
from trusthrl_sci.control.hierarchy import HierarchicalPolicy
from trusthrl_sci.control.ppo import MetaPPOUpdater, MetaRollout, PPOUpdater
from trusthrl_sci.control.rollout import RolloutBuffer
from trusthrl_sci.control.safety import PhaseLagrangian, aggregate_phase_costs, phase_budgets
from trusthrl_sci.runtime import atomic_torch_save
from trusthrl_sci.specification import ExperimentSpec

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingProgress:
    environment_steps: int
    episodes: int
    policy_updates: int
    meta_updates: int
    mean_reward: float
    mean_hrs: float
    mean_scr: float


class TrustHRLTrainer:
    def __init__(
        self,
        specification: ExperimentSpec,
        environment: InflammationEnvironment,
        policy: HierarchicalPolicy,
        policy_updater: PPOUpdater,
        meta_updater: MetaPPOUpdater,
        lagrangian: PhaseLagrangian,
        rollout: RolloutBuffer,
        policy_optimizer: torch.optim.Optimizer,
        meta_optimizer: torch.optim.Optimizer,
    ) -> None:
        self.specification = specification
        self.environment = environment
        self.policy = policy
        self.policy_updater = policy_updater
        self.meta_updater = meta_updater
        self.lagrangian = lagrangian
        self.rollout = rollout
        self.policy_optimizer = policy_optimizer
        self.meta_optimizer = meta_optimizer
        self.meta_rollout = MetaRollout()
        self.steps = 0
        self.episodes = 0
        self.policy_updates = 0
        self.meta_updates = 0

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "step": self.steps,
            "episode": self.episodes,
            "seed": self.specification.seed,
            "policy": self.policy.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "meta_optimizer": self.meta_optimizer.state_dict(),
            "lagrange": self.lagrangian.state_dict(),
            "specification": asdict(self.specification),
        }

    def save(self, path: str | Path) -> None:
        atomic_torch_save(self.checkpoint_payload(), path)

    def restore(self, payload: dict[str, object]) -> None:
        self.policy.load_state_dict(payload["policy"])
        self.policy_optimizer.load_state_dict(payload["policy_optimizer"])
        self.meta_optimizer.load_state_dict(payload["meta_optimizer"])
        self.lagrangian.load_state_dict(payload["lagrange"])
        self.steps = int(payload["step"])
        self.episodes = int(payload["episode"])

    def collect_episode(self) -> VectorEpisodeLedger:
        observation = self.environment.reset()
        ledger = VectorEpisodeLedger(self.environment.batch_size, self.environment.device)
        phase = self.environment.phase.clone()
        accumulated_meta_reward = torch.zeros_like(phase, dtype=torch.float32)
        for episode_step in range(self.specification.episode_steps):
            low_observation = observation.low_level(self.specification.severity_conditioning)
            with torch.no_grad():
                decision = self.policy.act(observation, phase, episode_step)
            if decision.phase_output is not None:
                if self.meta_rollout.observations:
                    terminal = torch.zeros_like(phase, dtype=torch.bool)
                    self.meta_rollout.append_return(accumulated_meta_reward, terminal)
                    accumulated_meta_reward.zero_()
                high_observation = observation.high_level(self.specification.severity_conditioning)
                self.meta_rollout.append_decision(
                    high_observation,
                    phase,
                    decision.phase_output.action,
                    decision.phase_output.log_probability,
                    decision.phase_output.value,
                )
            phase = self.environment.set_phase(decision.phase)
            output = self.environment.step(decision.dose)
            homeostatic = self.environment.basin_membership(output.state)
            ledger.update(output.reward, output.costs, decision.dose, homeostatic)
            accumulated_meta_reward += output.reward
            transition = Transition(
                observation=low_observation,
                action=decision.dose,
                log_probability=decision.dose_output.log_probability,
                reward=output.reward,
                costs=output.costs,
                value=decision.dose_output.value,
                terminated=output.terminated,
                phase=phase,
            )
            self.rollout.append(transition)
            self.steps += self.environment.batch_size
            observation = self.environment.observation()
            if self.rollout.full():
                self.optimize_rollout(observation, phase)
            if torch.all(output.terminated):
                self.meta_rollout.append_return(accumulated_meta_reward, output.terminated)
                break
        self.episodes += self.environment.batch_size
        self.optimize_meta()
        return ledger

    def optimize_rollout(self, observation: object, phase: Tensor) -> None:
        if not hasattr(observation, "low_level"):
            raise TypeError("observation does not expose low-level state")
        with torch.no_grad():
            last_value = self.policy.dose(observation, phase).value
        batch = self.rollout.batch(
            last_value,
            self.specification.discount,
            self.specification.gae_lambda,
        )
        report = self.policy_updater.update(batch)
        phase_costs = aggregate_phase_costs(
            self.rollout.costs[: self.rollout.position],
            self.rollout.phases[: self.rollout.position],
            self.specification.phase_count,
        )
        budgets = phase_budgets(
            self.rollout.phases[: self.rollout.position],
            self.specification.safety_budget,
            phase_count=self.specification.phase_count,
        )
        self.lagrangian.update(phase_costs, budgets)
        self.policy_updates += report.updates
        self.rollout.clear()
        LOGGER.info(
            "policy_update steps=%d loss=%.6f kl=%.6f",
            self.steps,
            report.total_loss,
            report.approximate_kl,
        )

    def optimize_meta(self) -> None:
        if not self.meta_rollout.observations:
            return
        if len(self.meta_rollout.rewards) != len(self.meta_rollout.observations):
            return
        report = self.meta_updater.update(self.meta_rollout)
        self.meta_updates += report.updates
        self.meta_rollout.clear()
        LOGGER.info("meta_update steps=%d loss=%.6f", self.steps, report.total_loss)

    def train(
        self,
        callback: Callable[[TrainingProgress], None] | None = None,
        checkpoint_interval: int = 50000,
        checkpoint_root: str | Path | None = None,
    ) -> TrainingProgress:
        latest = TrainingProgress(self.steps, self.episodes, 0, 0, 0, 0, 0)
        next_checkpoint = self.steps + checkpoint_interval
        while self.steps < self.specification.environment_steps:
            ledger = self.collect_episode()
            summaries = ledger.summaries()
            latest = TrainingProgress(
                environment_steps=self.steps,
                episodes=self.episodes,
                policy_updates=self.policy_updates,
                meta_updates=self.meta_updates,
                mean_reward=float(summaries["mean_reward"].mean()),
                mean_hrs=float(summaries["hrs"].mean()),
                mean_scr=float(summaries["scr"].mean()),
            )
            if callback is not None:
                callback(latest)
            if checkpoint_root is not None and self.steps >= next_checkpoint:
                path = Path(checkpoint_root) / f"step_{self.steps:09d}.pt"
                self.save(path)
                next_checkpoint += checkpoint_interval
        return latest
