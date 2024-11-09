import hydra
import numpy as np
import gymnasium as gym
import highway_env
from trainers.trainer import RefSpeedTrainer

from config.config import build_env_config, build_mpcrl_agent_config, build_pure_mpc_agent_config

@hydra.main(config_name="cfg", config_path="./config", version_base="1.3")
def train_mpcrl(cfg):
    
    gym_env_config = build_env_config(cfg)
    mpcrl_agent_config = build_mpcrl_agent_config(cfg)
    pure_mpc_agent_config = build_pure_mpc_agent_config(cfg)

    # env
    env = gym.make("intersection-v1", render_mode="rgb_array", config=gym_env_config)

    trainer = RefSpeedTrainer(env, mpcrl_agent_config, pure_mpc_agent_config)
    trainer.learn()
    trainer.save(path="weights", file="test")
            
if __name__ == "__main__":
    train_mpcrl()