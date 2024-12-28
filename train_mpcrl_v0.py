import hydra
import numpy as np
import gymnasium as gym
import highway_env
from trainers.trainer import RefSpeedTrainer

from config.config import build_env_config, build_mpcrl_agent_config, build_pure_mpc_agent_config

@hydra.main(config_name="cfg", config_path="./config", version_base="1.3")
def train_mpcrl(cfg):
    algorithm = cfg["mpc_rl"]["algorithm"]
    gym_env_config = build_env_config(cfg)
    mpcrl_agent_config = build_mpcrl_agent_config(cfg, version="v0")
    pure_mpc_agent_config = build_pure_mpc_agent_config(cfg)

    env = gym.make("intersection-v1", render_mode="rgb_array", config=gym_env_config)
    trainer = RefSpeedTrainer(env, mpcrl_agent_config, pure_mpc_agent_config)
    trainer.learn()
    trainer.save(path=f"./weights/v0", file=f"test_{algorithm}_v0")