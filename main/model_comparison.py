import os
import sys
import glob
import gymnasium as gym
import highway_env
import numpy as np
from collections import defaultdict
import torch
import hydra
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from agents.pure_mpc import PureMPC_Agent
from agents.pure_mpc_no_collision import PureMPC_Agent as PureMPC_NoCollision_Agent
from agents.ppo_agent import PPOAgent
from config.config import build_env_config, build_mpcrl_agent_config, build_pure_mpc_agent_config
from trainers.trainer import RefSpeedTrainer

# Evaluation Configuration
VISUALIZATION_MODE = False  # Set to False for multiple runs without visualization
N_EPISODES = 1 if VISUALIZATION_MODE else 100  # Number of episodes to run
RENDER_MODE = "human" if VISUALIZATION_MODE else "rgb_array"
MODELS_TO_EVALUATE = ["pure_mpc"]  # Options: ["pure_mpc", "pure_mpc_no_collision", "mpcrl", "ppo"]
MODELS_TO_EVALUATE = ["pure_mpc", "pure_mpc_no_collision", "mpcrl", "ppo"]
no_steps = 250

class ModelEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env_config = build_env_config(cfg)
        self.pure_mpc_cfg = build_pure_mpc_agent_config(cfg)
        self.pure_mpc_no_collision_cfg = build_pure_mpc_agent_config(cfg, use_collision_avoidance=False)
        self.mpcrl_cfg = build_mpcrl_agent_config(cfg, version="v0", algorithm="ppo")

    def evaluate_single_episode(self, model_type, model=None, env=None):
        """Run a single episode and return metrics"""
        observation, _ = env.reset()
        episode_steps = 0
        collisions = 0
        success = False
        total_speed = 0
        
        for step in range(no_steps):  # Run for 150 steps
            if model_type == "pure_mpc":
                action = model.predict(observation, return_numpy=True)
            elif model_type == "pure_mpc_no_collision":
                action = model.predict(observation, return_numpy=True)
            elif model_type == "mpcrl":
                action = model.predict(observation, False)
                action = [action.acceleration/5, action.steer/(np.pi/3)]
            elif model_type == "ppo":
                action = model.predict(observation)
                action = action.reshape(-1) if isinstance(action, np.ndarray) else action
            
            # Record speed before step
            current_speed = env.unwrapped.controlled_vehicles[0].speed
            total_speed += current_speed
            
            observation, reward, done, truncated, info = env.step(action)
            episode_steps += 1
            
            if VISUALIZATION_MODE:
                env.render()
                print(f"Step {step}, Speed: {current_speed:.2f} m/s, Reward: {reward}")
                if info.get("crashed", False):
                    print("COLLISION DETECTED!")
            
            # Check for collision
            if info.get("crashed", False):
                collisions += 1
            
            # Check for success (reaching the end) using the environment's has_arrived method
            if env.unwrapped.has_arrived(env.unwrapped.controlled_vehicles[0]):
                success = True
                if VISUALIZATION_MODE:
                    print("DESTINATION REACHED! Vehicle has arrived at exit.")
                break
                
            if done or truncated:
                break
        
        # Calculate metrics
        dt = 1 / env.unwrapped.config["policy_frequency"]  # Get dt from environment config
        travel_time = episode_steps * dt  # Convert steps to seconds
        avg_speed = total_speed / episode_steps if episode_steps > 0 else 0
        
        if VISUALIZATION_MODE:
            print(f"\nEpisode Statistics:")
            print(f"Average Speed: {avg_speed:.2f} m/s")
            print(f"Travel Time: {travel_time:.2f} seconds")
            print(f"Steps: {episode_steps}")
            print(f"Success: {'Yes' if success else 'No'}")
            print(f"Collisions: {'Yes' if collisions > 0 else 'No'}")
        
        return {
            "success": success,
            "collisions": collisions > 0,
            "steps": episode_steps,
            "avg_speed": avg_speed,
            "travel_time": travel_time
        }

    def run_evaluation(self):
        # Initialize results with default values
        results = {}
        for model in MODELS_TO_EVALUATE:
            results[model] = {
                "successes": 0,
                "collisions": 0,
                "total_steps": 0,
                "total_speed": 0,
                "total_time": 0,
                "episode_count": 0
            }
        
        # Initialize environment
        env = gym.make("intersection-v1", render_mode=RENDER_MODE, config=self.env_config)
        
        for model_type in MODELS_TO_EVALUATE:
            print(f"\nEvaluating {model_type}...")
            
            # Initialize the appropriate model
            if model_type == "pure_mpc":
                model = PureMPC_Agent(env, self.pure_mpc_cfg)
            elif model_type == "pure_mpc_no_collision":
                model = PureMPC_NoCollision_Agent(env, self.pure_mpc_no_collision_cfg)
            elif model_type == "mpcrl":
                model = RefSpeedTrainer(env, self.mpcrl_cfg, self.pure_mpc_cfg)
                # Find the latest saved model in the mpcrl subfolder
                save_dir = os.path.join(project_root, "saved_models", "mpcrl")
                model_files = sorted(glob.glob(f"{save_dir}/**/model_step_*.zip", recursive=True), 
                                  key=os.path.getmtime, reverse=True)
                if not model_files:
                    raise FileNotFoundError(f"No saved models found in {save_dir}")
                
                latest_model_path = model_files[0]
                print(f"Loading latest MPCRL model: {latest_model_path}")
                
                model.load(
                    path=latest_model_path,
                    mpcrl_cfg=self.mpcrl_cfg,
                    version="v0",
                    pure_mpc_cfg=self.pure_mpc_cfg,
                    env=env
                )
            elif model_type == "ppo":
                try:
                    model = PPOAgent(env, self.cfg)
                    latest_model_path = PPOAgent.get_latest_model_path()
                    print(f"Loading latest PPO model: {latest_model_path}")
                    model.load(latest_model_path)
                except FileNotFoundError as e:
                    print(f"Error loading PPO model: {e}")
                    continue
            
            # Run evaluation episodes
            for episode in range(N_EPISODES):
                print(f"Episode {episode + 1}/{N_EPISODES}")
                episode_results = self.evaluate_single_episode(model_type, model, env)
                
                # Update results
                results[model_type]["successes"] += int(episode_results["success"])
                results[model_type]["collisions"] += int(episode_results["collisions"])
                results[model_type]["total_steps"] += episode_results["steps"]
                results[model_type]["total_speed"] += episode_results["avg_speed"]
                results[model_type]["total_time"] += episode_results["travel_time"]
                results[model_type]["episode_count"] += 1
        
        env.close()
        return results

    def plot_results(self, results):
        if N_EPISODES == 1:
            return  # Don't create plots for single episodes
            
        # Define consistent model order for display
        model_display_order = ["pure_mpc", "pure_mpc_no_collision", "mpcrl", "ppo"]
        # Filter and sort models based on what's actually in the results
        models = [model for model in model_display_order if model in results]
        
        # Define readable labels for the plot
        model_labels = {
            "pure_mpc": "Pure MPC",
            "pure_mpc_no_collision": "MPC w/o Collision",
            "mpcrl": "MPC-RL",
            "ppo": "PPO"
        }
        
        metrics = {
            "Success Rate (%)": [results[m]["successes"]/results[m]["episode_count"] * 100 for m in models],
            "Collision Rate (%)": [results[m]["collisions"]/results[m]["episode_count"] * 100 for m in models],
            "Average Steps": [results[m]["total_steps"]/results[m]["episode_count"] for m in models],
            "Average Speed (m/s)": [results[m]["total_speed"]/results[m]["episode_count"] for m in models],
            "Average Time (s)": [results[m]["total_time"]/results[m]["episode_count"] for m in models]
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f', '#9b59b6']
        
        for i, (metric, values) in enumerate(metrics.items()):
            if i >= len(axes):
                break
            bars = axes[i].bar([model_labels[m] for m in models], values, color=colors[:len(models)])
            axes[i].set_title(metric)
            axes[i].set_xticklabels([model_labels[m] for m in models], rotation=45)
            
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.1f}',
                         ha='center', va='bottom')
        
        # Remove the extra subplot
        if len(axes) > len(metrics):
            axes[-1].remove()
        
        plt.tight_layout()
        plot_path = os.path.join(project_root, "model_comparison_results.png")
        plt.savefig(plot_path)
        print(f"\nResults plot saved to: {plot_path}")
        plt.close()

@hydra.main(config_name="cfg", config_path="../config", version_base="1.3")
def main(cfg):
    evaluator = ModelEvaluator(cfg)
    results = evaluator.run_evaluation()
    
    # Print results
    if not VISUALIZATION_MODE:
        print("\nEvaluation Results:")
        print("=" * 50)
        for model_type, metrics in results.items():
            n_episodes = metrics["episode_count"]
            print(f"\nModel: {model_type}")
            print(f"Success Rate: {metrics['successes']/n_episodes * 100:.1f}%")
            print(f"Collision Rate: {metrics['collisions']/n_episodes * 100:.1f}%")
            print(f"Average Steps: {metrics['total_steps']/n_episodes:.1f}")
            print(f"Average Speed: {metrics['total_speed']/n_episodes:.2f} m/s")
            print(f"Average Travel Time: {metrics['total_time']/n_episodes:.2f} seconds")
        
        # Plot results
        evaluator.plot_results(results)

if __name__ == "__main__":
    main()