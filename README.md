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

- [ ] `Config`: Add `hydra` and `YAML` configuration files for better organization of training and testing parameters.
- [ ] `MPC`:  Implement collision cost and vehicle dynamics updates for other agent vehicles
- [ ] `MPC+RL`: Integrate RL components into the `PureMPC_Agent` for enhanced decision-making in more complex scenarios.
- [ ] `Animation`:  Enhance the visualization by adding vehicle shapes and orientations to improve realism and interpretability.

## Completed

- [x] `MPC`:   Fixed the issue where the MPC agent did not follow the reference trajectory correctly.