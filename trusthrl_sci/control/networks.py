from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal

from trusthrl_sci.contracts import PolicyOutput


def orthogonal_initialize(module: nn.Module, gain: float = math.sqrt(2)) -> nn.Module:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.zeros_(module.bias)
    return module


class MLP(nn.Module):
    def __init__(
        self,
        input_width: int,
        output_width: int,
        hidden_width: int = 256,
        hidden_layers: int = 2,
        output_gain: float = 1.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_width
        for _ in range(hidden_layers):
            layer = nn.Linear(current, hidden_width)
            orthogonal_initialize(layer)
            layers.extend((layer, nn.ReLU()))
            current = hidden_width
        output = nn.Linear(current, output_width)
        orthogonal_initialize(output, output_gain)
        layers.append(output)
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class ContinuousActorCritic(nn.Module):
    def __init__(
        self,
        state_width: int,
        action_width: int,
        hidden_width: int = 256,
        hidden_layers: int = 2,
        initial_log_standard_deviation: float = -0.5,
    ) -> None:
        super().__init__()
        self.action_width = action_width
        self.actor = MLP(state_width, action_width, hidden_width, hidden_layers, 0.01)
        self.critic = MLP(state_width, 1, hidden_width, hidden_layers, 1.0)
        self.log_standard_deviation = nn.Parameter(
            torch.full((action_width,), initial_log_standard_deviation)
        )

    def distribution(self, observation: Tensor) -> Normal:
        mean = self.actor(observation)
        standard_deviation = self.log_standard_deviation.exp().expand_as(mean)
        return Normal(mean, standard_deviation)

    def squash(self, raw_action: Tensor) -> Tensor:
        return torch.sigmoid(raw_action)

    def unsquash(self, action: Tensor) -> Tensor:
        clipped = action.clamp(1e-6, 1 - 1e-6)
        return torch.logit(clipped)

    def corrected_log_probability(self, distribution: Normal, raw_action: Tensor) -> Tensor:
        action = self.squash(raw_action)
        correction = torch.log(action * (1 - action) + 1e-6)
        return (distribution.log_prob(raw_action) - correction).sum(dim=-1)

    def sample(self, observation: Tensor, deterministic: bool = False) -> PolicyOutput:
        distribution = self.distribution(observation)
        raw_action = distribution.mean if deterministic else distribution.rsample()
        action = self.squash(raw_action)
        log_probability = self.corrected_log_probability(distribution, raw_action)
        entropy = distribution.entropy().sum(dim=-1)
        value = self.critic(observation).squeeze(-1)
        return PolicyOutput(action, log_probability, entropy, value)

    def evaluate(self, observation: Tensor, action: Tensor) -> PolicyOutput:
        distribution = self.distribution(observation)
        raw_action = self.unsquash(action)
        log_probability = self.corrected_log_probability(distribution, raw_action)
        entropy = distribution.entropy().sum(dim=-1)
        value = self.critic(observation).squeeze(-1)
        return PolicyOutput(action, log_probability, entropy, value)


class DiscreteActorCritic(nn.Module):
    def __init__(
        self,
        state_width: int,
        choices: int = 3,
        hidden_width: int = 256,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.choices = choices
        self.actor = MLP(state_width, choices, hidden_width, hidden_layers, 0.01)
        self.critic = MLP(state_width, 1, hidden_width, hidden_layers, 1.0)

    def masked_logits(self, observation: Tensor, minimum: Tensor) -> Tensor:
        logits = self.actor(observation)
        indices = torch.arange(self.choices, device=observation.device)
        mask = indices.unsqueeze(0) < minimum.reshape(-1, 1)
        return logits.masked_fill(mask, torch.finfo(logits.dtype).min)

    def distribution(self, observation: Tensor, minimum: Tensor) -> Categorical:
        return Categorical(logits=self.masked_logits(observation, minimum))

    def sample(
        self,
        observation: Tensor,
        minimum: Tensor,
        deterministic: bool = False,
    ) -> PolicyOutput:
        distribution = self.distribution(observation, minimum)
        action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        log_probability = distribution.log_prob(action)
        entropy = distribution.entropy()
        value = self.critic(observation).squeeze(-1)
        return PolicyOutput(action, log_probability, entropy, value)

    def evaluate(self, observation: Tensor, minimum: Tensor, action: Tensor) -> PolicyOutput:
        distribution = self.distribution(observation, minimum)
        log_probability = distribution.log_prob(action)
        entropy = distribution.entropy()
        value = self.critic(observation).squeeze(-1)
        return PolicyOutput(action, log_probability, entropy, value)


@dataclass(frozen=True)
class NetworkDimensions:
    low_state: int
    high_state: int
    action: int
    hidden: int
    layers: int


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def gradient_norm(module: nn.Module) -> Tensor:
    gradients = [
        parameter.grad.detach().flatten()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.cat(gradients))
