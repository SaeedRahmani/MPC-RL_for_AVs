import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv

import hydra
import numpy as np
import gymnasium as gym
import highway_env
from trainers.trainer import RefSpeedTrainer
from stable_baselines3.common.vec_env import DummyVecEnv
from custom_models.custom_policy import CustomActorCriticPolicy
from custom_models.custom_features_extractor import CustomFeaturesExtractor  # Add this
from custom_models.observation_wrapper import StructuredObservationWrapper
from config.config import build_env_config, build_mpcrl_agent_config, build_pure_mpc_agent_config

@hydra.main(config_name="cfg", config_path="./config", version_base="1.3")
def train_mpcrl(cfg):
    # Specify algorithm directly here
    algorithm = "ppo" 
    
    gym_env_config = build_env_config(cfg)
    mpcrl_agent_config = build_mpcrl_agent_config(cfg, version="v0", algorithm=algorithm)
    pure_mpc_agent_config = build_pure_mpc_agent_config(cfg)
    
    env = gym.make("intersection-v1", render_mode="rgb_array", config=gym_env_config)
    env = StructuredObservationWrapper(env)
    env = DummyVecEnv([lambda: env])
    
    # Configure policy
    mpcrl_agent_config["policy_class"] = CustomActorCriticPolicy
    mpcrl_agent_config["policy_kwargs"] = {
        "features_extractor_class": CustomFeaturesExtractor,
        "features_extractor_kwargs": dict(features_dim=64)
    }
    
    trainer = RefSpeedTrainer(env, mpcrl_agent_config, pure_mpc_agent_config)
    trainer.learn()
    trainer.save(path=f"./weights", file=f"test_{algorithm}_v0")

if __name__ == "__main__":
    train_mpcrl()