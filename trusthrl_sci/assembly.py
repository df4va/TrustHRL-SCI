from __future__ import annotations

from dataclasses import dataclass

import torch

from trusthrl_sci.control.environment import InflammationEnvironment
from trusthrl_sci.control.hierarchy import HierarchicalPolicy
from trusthrl_sci.control.networks import NetworkDimensions
from trusthrl_sci.control.ppo import MetaPPOUpdater, PPOSettings, PPOUpdater
from trusthrl_sci.control.rollout import RolloutBuffer
from trusthrl_sci.control.safety import PhaseLagrangian
from trusthrl_sci.control.trainer import TrustHRLTrainer
from trusthrl_sci.dynamics.network import CytokineNetwork, ODEParameters, RK4Integrator
from trusthrl_sci.specification import ExperimentSpec


@dataclass(frozen=True)
class TrainingAssembly:
    network: CytokineNetwork
    integrator: RK4Integrator
    environment: InflammationEnvironment
    policy: HierarchicalPolicy
    lagrangian: PhaseLagrangian
    trainer: TrustHRLTrainer


def assemble_training(
    specification: ExperimentSpec,
    parameters: ODEParameters,
    device: torch.device,
    environments: int = 16,
) -> TrainingAssembly:
    parameters = parameters.to(device)
    network = CytokineNetwork(parameters, specification.kinetics).to(device)
    integrator = RK4Integrator(
        network,
        specification.rl_step_hours,
        specification.ode_substeps,
    ).to(device)
    environment = InflammationEnvironment(
        integrator=integrator,
        parameters=parameters,
        batch_size=environments,
        horizon=specification.horizon_hours,
        phase_boundaries=tuple(specification.phase_boundaries),
        immunodeficiency_floor=specification.immunodeficiency_floor,
        device=device,
    )
    dimensions = NetworkDimensions(
        low_state=specification.state_width,
        high_state=specification.meta_state_width,
        action=specification.actions,
        hidden=specification.hidden_width,
        layers=specification.hidden_layers,
    )
    policy = HierarchicalPolicy(
        dimensions,
        phases=specification.phase_count,
        meta_interval=specification.meta_interval_hours,
        severity_conditioning=specification.severity_conditioning,
    ).to(device)
    policy_optimizer = torch.optim.Adam(
        policy.subcontrollers.parameters(),
        lr=specification.learning_rate,
        betas=(specification.adam_beta1, specification.adam_beta2),
    )
    meta_optimizer = torch.optim.Adam(
        policy.meta.parameters(),
        lr=specification.learning_rate,
        betas=(specification.adam_beta1, specification.adam_beta2),
    )
    lagrangian = PhaseLagrangian(
        phases=specification.phase_count,
        constraints=3,
        learning_rate=specification.lagrange_learning_rate,
    ).to(device)
    settings = PPOSettings(
        clip_ratio=specification.clip_ratio,
        entropy_coefficient=specification.entropy_coefficient,
        value_coefficient=specification.value_coefficient,
        gradient_clip=specification.gradient_clip,
        epochs=specification.ppo_epochs,
        minibatch_size=specification.minibatch_size,
    )
    policy_updater = PPOUpdater(policy, policy_optimizer, settings, lagrangian)
    meta_updater = MetaPPOUpdater(policy, meta_optimizer, settings, specification.discount)
    rollout_length = specification.rollout_size // environments
    rollout = RolloutBuffer(
        capacity=rollout_length,
        environments=environments,
        observation_width=specification.state_width,
        action_width=specification.actions,
        constraints=3,
        device=device,
    )
    trainer = TrustHRLTrainer(
        specification,
        environment,
        policy,
        policy_updater,
        meta_updater,
        lagrangian,
        rollout,
        policy_optimizer,
        meta_optimizer,
    )
    return TrainingAssembly(network, integrator, environment, policy, lagrangian, trainer)
