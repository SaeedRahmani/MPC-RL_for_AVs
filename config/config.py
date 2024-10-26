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

def build_pure_mpc_agent_config(cfg) -> dict:
    return cfg["pure_mpc"]