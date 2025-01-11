import numpy as np


def build_env_config(cfg) -> dict:
    """
    Build the configuration used for Env `Highway-env:Intersection-v1`
    """
    env_config = cfg["env"]
    return {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": env_config["vehicles_count"],
            "features": ["presence", "x", "y", "vx", "vy", "heading", "sin_h", "cos_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20],
                "heading": [-1 * np.pi, np.pi],
                "sin_h": [-1, 1],
                "cos_h": [-1, 1],
            },
            "order": env_config["order"],
            "absolute": env_config["absolute"],
            "normalize": env_config["normalize"],
        },
        "action": {
            "type": "ContinuousAction",
            "steering_range": [-np.pi / 4, np.pi / 4],
            "acceleration_range": [-5.0, 5.0],
            "longitudinal": True,
            "lateral": True,
            "dynamical": True,
        },
        "scaling": 3, # scale the rendering animation to show all the surrounding vehicles.
        "duration": 10,  # [s]
        # "destination": "o1",
        "initial_vehicle_count": env_config["initial_vehicle_count"],
        "spawn_probability": env_config["spawn_probability"],
        "screen_width": 600,
        "screen_height": 600,
        "policy_frequency": 10,
        "simulation_frequency": 30
    }

# def build_mpcrl_agent_config(cfg, version: str = "v0") -> dict:
#     # return cfg["mpc_rl"][version]
#     # Changed to work with the new config file
#     """Build MPC-RL agent configuration based on algorithm and version."""
#     algorithm = cfg["mpc_rl"]["algorithm"]
#     # Convert OmegaConf to dict and add algorithm
#     config = dict(cfg["mpc_rl"][algorithm][version])
#     # Add algorithm in new dict
#     return {
#         **config,
#         "algorithm": algorithm
#     }

def build_mpcrl_agent_config(cfg, version: str = "v0", algorithm: str = "ppo") -> dict:
    # Override algorithm in config
    cfg.mpc_rl.algorithm = algorithm
    print('algorithm in config.py:', algorithm)
    config = dict(cfg["mpc_rl"][algorithm][version])
    config["algorithm"] = algorithm
    return config

def build_pure_mpc_agent_config(cfg, use_collision_avoidance: bool = True) -> dict:
    """Build MPC agent configuration with option to choose version.
    
    Args:
        cfg: The main config dictionary
        use_collision_avoidance: If True, use original MPC with collision avoidance.
                                If False, use simplified MPC without collision avoidance.
    """
    if use_collision_avoidance:
        return cfg["pure_mpc"]
    else:
        return cfg["pure_mpc_no_collision"]