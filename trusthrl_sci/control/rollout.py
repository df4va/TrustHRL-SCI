from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from trusthrl_sci.contracts import RolloutBatch, Transition


@dataclass(frozen=True)
class AdvantageResult:
    reward_advantages: Tensor
    returns: Tensor
    cost_advantages: Tensor


class RolloutBuffer:
    def __init__(
        self,
        capacity: int,
        environments: int,
        observation_width: int,
        action_width: int,
        constraints: int,
        device: torch.device,
    ) -> None:
        self.capacity = capacity
        self.environments = environments
        self.device = device
        self.observations = torch.empty((capacity, environments, observation_width), device=device)
        self.actions = torch.empty((capacity, environments, action_width), device=device)
        self.log_probabilities = torch.empty((capacity, environments), device=device)
        self.rewards = torch.empty((capacity, environments), device=device)
        self.costs = torch.empty((capacity, environments, constraints), device=device)
        self.values = torch.empty((capacity, environments), device=device)
        self.terminated = torch.empty((capacity, environments), dtype=torch.bool, device=device)
        self.phases = torch.empty((capacity, environments), dtype=torch.long, device=device)
        self.position = 0

    def clear(self) -> None:
        self.position = 0

    def append(self, transition: Transition) -> None:
        if self.position >= self.capacity:
            raise RuntimeError("rollout buffer is full")
        index = self.position
        self.observations[index].copy_(transition.observation)
        self.actions[index].copy_(transition.action)
        self.log_probabilities[index].copy_(transition.log_probability)
        self.rewards[index].copy_(transition.reward)
        self.costs[index].copy_(transition.costs)
        self.values[index].copy_(transition.value)
        self.terminated[index].copy_(transition.terminated)
        self.phases[index].copy_(transition.phase)
        self.position += 1

    def full(self) -> bool:
        return self.position == self.capacity

    def advantages(
        self,
        last_value: Tensor,
        discount: float,
        gae_lambda: float,
    ) -> AdvantageResult:
        length = self.position
        reward_advantages = torch.zeros_like(self.rewards[:length])
        cost_advantages = torch.zeros_like(self.costs[:length])
        reward_accumulator = torch.zeros(self.environments, device=self.device)
        cost_accumulator = torch.zeros(
            (self.environments, self.costs.shape[-1]), device=self.device
        )
        next_value = last_value
        next_cost_value = torch.zeros_like(cost_accumulator)
        for index in range(length - 1, -1, -1):
            continuation = 1 - self.terminated[index].float()
            reward_delta = (
                self.rewards[index] + discount * continuation * next_value - self.values[index]
            )
            reward_accumulator = (
                reward_delta + discount * gae_lambda * continuation * reward_accumulator
            )
            reward_advantages[index] = reward_accumulator
            cost_delta = self.costs[index] + discount * continuation.unsqueeze(-1) * next_cost_value
            cost_accumulator = (
                cost_delta + discount * gae_lambda * continuation.unsqueeze(-1) * cost_accumulator
            )
            cost_advantages[index] = cost_accumulator
            next_value = self.values[index]
            next_cost_value = self.costs[index]
        returns = reward_advantages + self.values[:length]
        return AdvantageResult(reward_advantages, returns, cost_advantages)

    def batch(self, last_value: Tensor, discount: float, gae_lambda: float) -> RolloutBatch:
        result = self.advantages(last_value, discount, gae_lambda)
        length = self.position
        advantages = result.reward_advantages.reshape(-1)
        normalized = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
        cost_advantages = result.cost_advantages.reshape(-1, self.costs.shape[-1])
        return RolloutBatch(
            observations=self.observations[:length].reshape(-1, self.observations.shape[-1]),
            actions=self.actions[:length].reshape(-1, self.actions.shape[-1]),
            old_log_probabilities=self.log_probabilities[:length].reshape(-1),
            advantages=normalized,
            returns=result.returns.reshape(-1),
            cost_advantages=cost_advantages,
            phases=self.phases[:length].reshape(-1),
        )


def minibatches(
    batch: RolloutBatch,
    size: int,
    generator: torch.Generator | None = None,
) -> Iterator[RolloutBatch]:
    count = batch.size()
    order = torch.randperm(count, device=batch.observations.device, generator=generator)
    for start in range(0, count, size):
        indices = order[start : start + size]
        yield RolloutBatch(
            observations=batch.observations[indices],
            actions=batch.actions[indices],
            old_log_probabilities=batch.old_log_probabilities[indices],
            advantages=batch.advantages[indices],
            returns=batch.returns[indices],
            cost_advantages=batch.cost_advantages[indices],
            phases=batch.phases[indices],
        )
