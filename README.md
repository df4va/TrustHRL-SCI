# Trustworthy Hierarchical Reinforcement Learning for Spinal Cord Injury

TrustHRL controls a calibrated 28-cytokine dynamical system with a monotonic three-phase hierarchy and phase-specific constrained optimization. The implementation covers the coupled Hill system, severity conditioning, fourth-order Runge–Kutta integration, hierarchical PPO, phase-level Lagrange updates, the four reported outcome metrics, baseline controllers, ablations, cross-validation, and statistical comparisons.

## Scope

The control horizon is 168 hours. Acute, subacute, and chronic decisions occupy 0–6, 6–72, and 72–168 hours. The meta-controller acts every 6 hours. Phase-specific sub-controllers act hourly while the ODE uses ten 0.1-hour substeps. Three costs monitor cytokine ceilings, rolling 24-hour dose limits, and the 5% pro-inflammatory floor.

The calibrated 28×28 interaction matrix, Hill constants, production and decay rates, drug distribution matrix, equilibria, and safety ceilings are required inputs. They are not embedded because the manuscript does not enumerate their numeric values. Supplying invented values would not yield the reported experiment.

## Installation

The reference environment uses Python 3.10.13, PyTorch 2.1.2, CUDA 11.8, and the pinned packages in `requirements.txt`.

```bash
conda env create -f environment.yml
conda activate trusthrl-sci
python -m pip install --no-deps .
```

The container image is built with:

```bash
docker build -t trusthrl-sci .
```

## Data

Verified accession and article URLs are collected in `datasets.txt`. GEO data can be obtained with `GEODownloader`; preprocessing fits min–max and z-score statistics on the training partition only. Transcriptomic validation uses training-fitted log2 fold change transforms.

Official accession metadata differ from two manuscript table entries. GSE5296 is registered as a mouse microarray series with 96 samples. GSE151371 is registered with 58 samples. The loaders follow official accession metadata while retaining the manuscript’s intended experimental roles.

Prepare a calibration archive containing `severities`, `times`, `observations`, and `masks`, then fit the ODE parameters:

```bash
trusthrl-calibrate --template initial_parameters.npz --series cytokine_series.npz --output datasets/calibration/ode_parameters.npz
```

## Training

The primary configuration preserves the reported settings: 500,000 environment steps per seed, three seeds, rollout batch 2,048, minibatch 64, ten PPO epochs, Adam learning rate 3e-4, discount 0.99, GAE 0.95, clip ratio 0.2, entropy coefficient 0.01, and Lagrange learning rate 0.01.

```bash
trusthrl-train --config presets/main.yaml --parameters datasets/calibration/ode_parameters.npz --environments 16 --output artifacts/main
```

Run each ablation by replacing the configuration path with the corresponding file in `presets`. Configuration inheritance changes only the named factor.

## Evaluation

Trajectory archives contain controlled and uncontrolled states, homeostatic and initial states, actions, and the three cost channels.

```bash
trusthrl-evaluate --trajectories artifacts/main/trajectories.npz --output artifacts/main/metrics.json
```

Primary targets over three seeds are HRS 73.4 ± 2.1%, SCR 97.2 ± 1.1%, pro-inflammatory overshoot reduction 53.8 ± 2.3%, and treatment efficiency 0.29 ± 0.02 on the Hellenbrand atlas. `assessment.reference` records every reported primary and ablation value for tolerance checks.

## Compute budget

The manuscript reports one NVIDIA A100 GPU, 2.8 GB peak device memory, 6.8 hours for 500,000 TrustHRL environment steps, and 2.1 ms CPU action latency. Three primary seeds require approximately 20.4 A100 GPU-hours. The full baseline, sensitivity, cross-validation, and ablation matrix requires substantially more compute. Dataset storage is dominated by the GEO raw archives and should reserve at least 10 GB before extraction.

## Numerical and reporting limits

The manuscript specifies monotonic cubic interpolation across four severity levels but omits spline boundary conditions. Calibration is exposed as an explicit stage; runtime parameter selection uses deterministic interpolation. The paper also omits the reward threshold used to convert normalized reward to HRS, the eight intervention identities and units, numeric safety ceilings, dose limits, calibrated equilibria, and complete ODE parameter tables. These values must be supplied with the calibration archive before reported numbers can be claimed.

GEO records do not assign a single dataset-wide software-style license. Their public status and NCBI terms do not transfer rights in third-party submitted content. Users remain responsible for the terms attached to each record and source publication.

This software is for computational research. It does not produce clinical treatment recommendations.
