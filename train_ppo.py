import os
import hydra
import gymnasium as gym
import highway_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from agents.ppo_agent import PPOAgent
from config.config import build_env_config

@hydra.main(config_name="cfg", config_path="./config", version_base="1.3")
def train_ppo(cfg):
    # Build environment config
    env_config = build_env_config(cfg)
    
    def make_env():
        def _init():
            env = gym.make(
                "intersection-v1",
                render_mode=None,
                config=env_config
            )
            return env
        return _init

    # Create vectorized environment
    env = DummyVecEnv([make_env()])
    env = VecMonitor(env, "logs/ppo_monitor.csv")
    
    # Create and train agent
    agent = PPOAgent(env, cfg)
    agent.learn(save_dir="./saved_models")
    
    # Save final model
    final_model_path = os.path.join("saved_models", "ppo_final")
    agent.save(final_model_path)
    
    env.close()

if __name__ == "__main__":
    train_ppo()