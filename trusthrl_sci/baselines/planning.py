from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from trusthrl_sci.dynamics.network import RK4Integrator


@dataclass(frozen=True)
class MPCSettings:
    horizon: int = 12
    iterations: int = 100
    learning_rate: float = 1.0
    dose_penalty: float = 0.01
    smoothness_penalty: float = 0.01
    safety_penalty: float = 10.0


class ModelPredictiveController:
    def __init__(
        self,
        integrator: RK4Integrator,
        action_width: int,
        settings: MPCSettings,
        safety_ceiling: Tensor,
    ) -> None:
        self.integrator = integrator
        self.action_width = action_width
        self.settings = settings
        self.safety_ceiling = safety_ceiling
        self.previous_plan: Tensor | None = None

    def reset(self, batch_size: int, device: torch.device) -> None:
        self.previous_plan = torch.zeros(
            (batch_size, self.settings.horizon, self.action_width), device=device
        )

    def rollout(self, state: Tensor, actions: Tensor, severity: Tensor) -> Tensor:
        current = state
        trajectory = []
        for step in range(self.settings.horizon):
            current = self.integrator(current, actions[:, step], severity)
            trajectory.append(current)
        return torch.stack(trajectory, dim=1)

    def objective(
        self,
        state: Tensor,
        raw_actions: Tensor,
        severity: Tensor,
        target: Tensor,
    ) -> Tensor:
        actions = torch.sigmoid(raw_actions)
        trajectory = self.rollout(state, actions, severity)
        tracking = torch.square(trajectory - target.unsqueeze(1)).mean(dim=(1, 2))
        dose = actions.sum(dim=(1, 2))
        differences = actions[:, 1:] - actions[:, :-1]
        smoothness = torch.square(differences).mean(dim=(1, 2))
        violations = torch.relu(trajectory - self.safety_ceiling.reshape(1, 1, -1))
        safety = torch.square(violations).mean(dim=(1, 2))
        return (
            tracking
            + self.settings.dose_penalty * dose
            + self.settings.smoothness_penalty * smoothness
            + self.settings.safety_penalty * safety
        ).sum()

    def action(self, state: Tensor, target: Tensor, severity: Tensor) -> Tensor:
        if self.previous_plan is None:
            raise RuntimeError("controller must be reset")
        shifted = torch.cat((self.previous_plan[:, 1:], self.previous_plan[:, -1:]), dim=1).clamp(
            1e-5, 1 - 1e-5
        )
        raw_actions = torch.logit(shifted).detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [raw_actions],
            lr=self.settings.learning_rate,
            max_iter=self.settings.iterations,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = self.objective(state, raw_actions, severity, target)
            loss.backward()
            return loss

        optimizer.step(closure)
        plan = torch.sigmoid(raw_actions).detach()
        self.previous_plan.copy_(plan)
        return plan[:, 0]


def evaluate_controller(
    controller: object,
    integrator: RK4Integrator,
    initial: Tensor,
    target: Tensor,
    severity: Tensor,
    horizon: int,
) -> tuple[Tensor, Tensor]:
    if not hasattr(controller, "action"):
        raise TypeError("controller lacks action method")
    state = initial
    states = [state]
    actions = []
    for step in range(horizon):
        time = torch.full((state.shape[0],), step, device=state.device)
        action = controller.action(state, target, time)
        state = integrator(state, action, severity)
        states.append(state)
        actions.append(action)
    return torch.stack(states, dim=1), torch.stack(actions, dim=1)
