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

## Structure

```
tree 

│   .gitattributes
│   .gitignore
│   README.md
│   requirements.txt
│   setup.py
│
├───agents
│   │   a2c_mpc.py
│   │   base.py
│   │   ppo_mpc.py
│   │   pure_mpc.py
│   │   utils.py
│   │   __init__.py
│
├───config
│   │   cfg.yaml
│   │   config.py
│   │   __init__.py
│
├───main
│   │   run_pure_mpc.py
│   │   train_a2c_mpc.py
│   │
│   └───test_functionality
│           test_sb3.py
│           test_traj.py
│
└───trainers
    │   trainer.py
    │   utils.py
    │   __init__.py
```

## To-do List

- [ ] `Config`: Make a full configuration file and output structure, loggings.
- [ ] `MPC`: generate new reference speed if collision is detected.
- [ ] `MPC`:  Adapt collision cost and vehicle dynamics updates for other agent vehicles.
- [ ] `Animation`:  Enhance the visualization by adding vehicle shapes and orientations.

## Completed

- [x] `MPC+RL`: Integrate `PureMPC_Agent` as a component into `stable_baselines3` agents.
- [x] `RL`: Add stable_baselines as dependencies.
- [x] `Config`: Add `hydra` and `YAML` configuration files for better organization of training and testing parameters.
- [x] `MPC`:   Fixed the issue where the MPC agent did not follow the reference trajectory correctly.