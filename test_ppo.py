import os
import hydra
import gymnasium as gym
import highway_env

from agents.ppo_agent import PPOAgent
from config.config import build_env_config

@hydra.main(config_name="cfg", config_path="./config", version_base="1.3")
def test_ppo(cfg):
    # Build environment config
    env_config = build_env_config(cfg)
    
    # Create environment
    env = gym.make("intersection-v1", render_mode="human", config=env_config)
    
    # Initialize agent
    agent = PPOAgent(env, cfg)
    
    try:
        # Find and load latest model
        latest_model_path = PPOAgent.get_latest_model_path()
        print(f"Loading latest model: {latest_model_path}")
        agent.load(latest_model_path)
        
        # Test the model
        for episode in range(5):  # Test for 5 episodes
            print(f"\nTesting Episode {episode + 1}")
            observation, _ = env.reset()
            episode_reward = 0
            step = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action = agent.predict(observation)
                observation, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step += 1
                
                if info.get("crashed", False):
                    print(f"Collision at step {step}!")
                if info.get("arrived", False):
                    print(f"Successfully arrived at destination at step {step}!")
            
            print(f"Episode finished after {step} steps with reward {episode_reward}")
    
    finally:
        env.close()

if __name__ == "__main__":
    test_ppo()