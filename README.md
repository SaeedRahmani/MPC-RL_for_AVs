# `MPC*RL`

A framework for integrating Model Predictive Control (MPC) and Single-agent Reinforcement Learning (RL) for autonomous vehicle control in complex unsignalized intersection driving environments. The primary focus is on optimizing vehicle trajectories and control strategies using MPC, with future extensions for RL enhancements.

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/mpcrl.git
cd mpcrl
```

Install in the develop mode:

```bash
pip install -e .
# pip install -r requirements.txt
```


## To-do List

- [ ] `RL`: Add stable_baselines as dependencies.
- [ ] `Config`: Make a full configuration file and output structure, loggings.
- [ ] `MPC`: generate new reference speed if collision is detected.
- [ ] `MPC+RL`: Integrate RL components into the `PureMPC_Agent` for enhanced decision-making in more complex scenarios.
- [ ] `MPC`:  Adapt collision cost and vehicle dynamics updates for other agent vehicles
- [ ] `Animation`:  Enhance the visualization by adding vehicle shapes and orientations to improve realism and interpretability.

## Completed

- [x] `Config`: Add `hydra` and `YAML` configuration files for better organization of training and testing parameters.
- [x] `MPC`:   Fixed the issue where the MPC agent did not follow the reference trajectory correctly.