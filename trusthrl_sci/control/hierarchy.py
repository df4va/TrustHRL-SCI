from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from trusthrl_sci.contracts import Observation, PolicyOutput
from trusthrl_sci.control.networks import (
    ContinuousActorCritic,
    DiscreteActorCritic,
    NetworkDimensions,
)


@dataclass(frozen=True)
class HierarchicalAction:
    phase: Tensor
    dose: Tensor
    phase_output: PolicyOutput | None
    dose_output: PolicyOutput


class HierarchicalPolicy(nn.Module):
    def __init__(
        self,
        dimensions: NetworkDimensions,
        phases: int = 3,
        meta_interval: int = 6,
        severity_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.phases = phases
        self.meta_interval = meta_interval
        self.severity_conditioning = severity_conditioning
        self.meta = DiscreteActorCritic(
            dimensions.high_state,
            phases,
            dimensions.hidden,
            dimensions.layers,
        )
        self.subcontrollers = nn.ModuleList(
            ContinuousActorCritic(
                dimensions.low_state,
                dimensions.action,
                dimensions.hidden,
                dimensions.layers,
            )
            for _ in range(phases)
        )

    def decide_phase(
        self,
        observation: Observation,
        current_phase: Tensor,
        deterministic: bool = False,
    ) -> PolicyOutput:
        high_state = observation.high_level(self.severity_conditioning)
        return self.meta.sample(high_state, current_phase, deterministic)

    def dose(
        self,
        observation: Observation,
        phase: Tensor,
        deterministic: bool = False,
    ) -> PolicyOutput:
        low_state = observation.low_level(self.severity_conditioning)
        batch = low_state.shape[0]
        action_width = self.subcontrollers[0].action_width
        action = torch.empty((batch, action_width), device=low_state.device)
        log_probability = torch.empty(batch, device=low_state.device)
        entropy = torch.empty(batch, device=low_state.device)
        value = torch.empty(batch, device=low_state.device)
        for index, controller in enumerate(self.subcontrollers):
            mask = phase == index
            if not torch.any(mask):
                continue
            output = controller.sample(low_state[mask], deterministic)
            action[mask] = output.action
            log_probability[mask] = output.log_probability
            entropy[mask] = output.entropy
            value[mask] = output.value
        return PolicyOutput(action, log_probability, entropy, value)

    def act(
        self,
        observation: Observation,
        current_phase: Tensor,
        step: int,
        deterministic: bool = False,
    ) -> HierarchicalAction:
        phase_output = None
        phase = current_phase
        if step % self.meta_interval == 0:
            phase_output = self.decide_phase(observation, current_phase, deterministic)
            phase = torch.maximum(current_phase, phase_output.action.long())
        dose_output = self.dose(observation, phase, deterministic)
        return HierarchicalAction(phase, dose_output.action, phase_output, dose_output)

    def evaluate_doses(self, observations: Tensor, phases: Tensor, actions: Tensor) -> PolicyOutput:
        batch = observations.shape[0]
        log_probability = torch.empty(batch, device=observations.device)
        entropy = torch.empty(batch, device=observations.device)
        value = torch.empty(batch, device=observations.device)
        for index, controller in enumerate(self.subcontrollers):
            mask = phases == index
            if not torch.any(mask):
                continue
            output = controller.evaluate(observations[mask], actions[mask])
            log_probability[mask] = output.log_probability
            entropy[mask] = output.entropy
            value[mask] = output.value
        return PolicyOutput(actions, log_probability, entropy, value)


class FlatConstrainedPolicy(nn.Module):
    def __init__(self, dimensions: NetworkDimensions) -> None:
        super().__init__()
        self.policy = ContinuousActorCritic(
            dimensions.low_state,
            dimensions.action,
            dimensions.hidden,
            dimensions.layers,
        )

    def act(self, observation: Observation, deterministic: bool = False) -> PolicyOutput:
        return self.policy.sample(observation.low_level(), deterministic)
