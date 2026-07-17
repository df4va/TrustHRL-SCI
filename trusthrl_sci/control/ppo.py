from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from trusthrl_sci.contracts import RolloutBatch
from trusthrl_sci.control.hierarchy import HierarchicalPolicy
from trusthrl_sci.control.rollout import minibatches
from trusthrl_sci.control.safety import PhaseLagrangian


@dataclass(frozen=True)
class PPOSettings:
    clip_ratio: float
    entropy_coefficient: float
    value_coefficient: float
    gradient_clip: float
    epochs: int
    minibatch_size: int


@dataclass(frozen=True)
class PPOLoss:
    total: Tensor
    policy: Tensor
    value: Tensor
    entropy: Tensor
    constraint: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor


@dataclass(frozen=True)
class PPOUpdateReport:
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    constraint_loss: float
    approximate_kl: float
    clip_fraction: float
    updates: int


class PPOObjective:
    def __init__(self, settings: PPOSettings, lagrangian: PhaseLagrangian) -> None:
        self.settings = settings
        self.lagrangian = lagrangian

    def __call__(
        self,
        new_log_probability: Tensor,
        old_log_probability: Tensor,
        advantages: Tensor,
        new_value: Tensor,
        returns: Tensor,
        entropy: Tensor,
        cost_advantages: Tensor,
        phases: Tensor,
    ) -> PPOLoss:
        log_ratio = new_log_probability - old_log_probability
        ratio = log_ratio.exp()
        unclipped = ratio * advantages
        clipped_ratio = ratio.clamp(1 - self.settings.clip_ratio, 1 + self.settings.clip_ratio)
        clipped = clipped_ratio * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        value_loss = 0.5 * torch.square(new_value - returns).mean()
        entropy_mean = entropy.mean()
        penalty = self.lagrangian.penalty(cost_advantages, phases)
        constraint_loss = (ratio * penalty).mean()
        total = (
            policy_loss
            + self.settings.value_coefficient * value_loss
            - self.settings.entropy_coefficient * entropy_mean
            + constraint_loss
        )
        approximate_kl = ((ratio - 1) - log_ratio).mean()
        clip_fraction = ((ratio - 1).abs() > self.settings.clip_ratio).float().mean()
        return PPOLoss(
            total,
            policy_loss,
            value_loss,
            entropy_mean,
            constraint_loss,
            approximate_kl,
            clip_fraction,
        )


class PPOUpdater:
    def __init__(
        self,
        policy: HierarchicalPolicy,
        optimizer: torch.optim.Optimizer,
        settings: PPOSettings,
        lagrangian: PhaseLagrangian,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.settings = settings
        self.objective = PPOObjective(settings, lagrangian)

    def update(self, batch: RolloutBatch) -> PPOUpdateReport:
        totals = torch.zeros(7, device=batch.observations.device)
        updates = 0
        for _ in range(self.settings.epochs):
            for mini in minibatches(batch, self.settings.minibatch_size):
                output = self.policy.evaluate_doses(
                    mini.observations,
                    mini.phases,
                    mini.actions,
                )
                loss = self.objective(
                    output.log_probability,
                    mini.old_log_probabilities,
                    mini.advantages,
                    output.value,
                    mini.returns,
                    output.entropy,
                    mini.cost_advantages,
                    mini.phases,
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.total.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.settings.gradient_clip)
                self.optimizer.step()
                totals += torch.stack(
                    (
                        loss.total.detach(),
                        loss.policy.detach(),
                        loss.value.detach(),
                        loss.entropy.detach(),
                        loss.constraint.detach(),
                        loss.approximate_kl.detach(),
                        loss.clip_fraction.detach(),
                    )
                )
                updates += 1
        means = (totals / max(updates, 1)).cpu().tolist()
        return PPOUpdateReport(*means, updates)


class MetaRollout:
    def __init__(self) -> None:
        self.observations: list[Tensor] = []
        self.minimum_phases: list[Tensor] = []
        self.actions: list[Tensor] = []
        self.log_probabilities: list[Tensor] = []
        self.values: list[Tensor] = []
        self.rewards: list[Tensor] = []
        self.terminated: list[Tensor] = []

    def append_decision(
        self,
        observation: Tensor,
        minimum_phase: Tensor,
        action: Tensor,
        log_probability: Tensor,
        value: Tensor,
    ) -> None:
        self.observations.append(observation)
        self.minimum_phases.append(minimum_phase)
        self.actions.append(action)
        self.log_probabilities.append(log_probability)
        self.values.append(value)

    def append_return(self, reward: Tensor, terminated: Tensor) -> None:
        self.rewards.append(reward)
        self.terminated.append(terminated)

    def clear(self) -> None:
        self.observations.clear()
        self.minimum_phases.clear()
        self.actions.clear()
        self.log_probabilities.clear()
        self.values.clear()
        self.rewards.clear()
        self.terminated.clear()


class MetaPPOUpdater:
    def __init__(
        self,
        policy: HierarchicalPolicy,
        optimizer: torch.optim.Optimizer,
        settings: PPOSettings,
        discount: float,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.settings = settings
        self.discount = discount

    def returns(self, rollout: MetaRollout) -> Tensor:
        accumulator = torch.zeros_like(rollout.rewards[-1])
        values = []
        for reward, terminated in zip(
            reversed(rollout.rewards), reversed(rollout.terminated), strict=True
        ):
            accumulator = reward + self.discount * (1 - terminated.float()) * accumulator
            values.append(accumulator)
        return torch.stack(list(reversed(values)))

    def update(self, rollout: MetaRollout) -> PPOUpdateReport:
        observations = torch.stack(rollout.observations)
        minimum = torch.stack(rollout.minimum_phases)
        actions = torch.stack(rollout.actions)
        old_log_probability = torch.stack(rollout.log_probabilities)
        old_values = torch.stack(rollout.values)
        returns = self.returns(rollout)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
        totals = torch.zeros(7, device=observations.device)
        updates = 0
        for _ in range(self.settings.epochs):
            output = self.policy.meta.evaluate(
                observations.reshape(-1, observations.shape[-1]),
                minimum.reshape(-1),
                actions.reshape(-1),
            )
            new_log_probability = output.log_probability.reshape_as(old_log_probability)
            ratio = (new_log_probability - old_log_probability).exp()
            unclipped = ratio * advantages
            clipped = (
                ratio.clamp(
                    1 - self.settings.clip_ratio,
                    1 + self.settings.clip_ratio,
                )
                * advantages
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * torch.square(output.value.reshape_as(returns) - returns).mean()
            entropy = output.entropy.mean()
            total = (
                policy_loss
                + self.settings.value_coefficient * value_loss
                - self.settings.entropy_coefficient * entropy
            )
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(self.policy.meta.parameters(), self.settings.gradient_clip)
            self.optimizer.step()
            log_ratio = new_log_probability - old_log_probability
            approximate_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1).abs() > self.settings.clip_ratio).float().mean()
            totals += torch.stack(
                (
                    total.detach(),
                    policy_loss.detach(),
                    value_loss.detach(),
                    entropy.detach(),
                    torch.zeros((), device=total.device),
                    approximate_kl.detach(),
                    clip_fraction.detach(),
                )
            )
            updates += 1
        means = (totals / max(updates, 1)).cpu().tolist()
        return PPOUpdateReport(*means, updates)
