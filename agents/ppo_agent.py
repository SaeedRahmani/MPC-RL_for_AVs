from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os
import gymnasium as gym

class PPOAgent:
    def __init__(self, env, cfg):
        """
        Initialize PPO agent.
        
        Args:
            env: The training environment
            cfg: Configuration dictionary from hydra
        """
        self.env = env
        self.ppo_cfg = cfg["ppo"]
        
        # Initialize model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.ppo_cfg["learning_rate"],
            n_steps=self.ppo_cfg["n_steps"],
            batch_size=self.ppo_cfg["batch_size"],
            n_epochs=self.ppo_cfg["n_epochs"],
            gamma=self.ppo_cfg["gamma"],
            gae_lambda=self.ppo_cfg["gae_lambda"],
            clip_range=self.ppo_cfg["clip_range"],
            ent_coef=self.ppo_cfg["ent_coef"],
            verbose=1
        )

    def setup_save_directory(self, base_dir="saved_models"):
        """Create a new subfolder for this training run."""
        import datetime
        # Create main PPO directory if it doesn't exist
        ppo_dir = os.path.join(base_dir, "ppo")
        os.makedirs(ppo_dir, exist_ok=True)
        
        # Create timestamped subfolder for this run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(ppo_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)
        
        return save_dir

    def learn(self, save_dir="saved_models"):
        """Train the PPO agent."""
        # Setup save directory
        model_dir = self.setup_save_directory(save_dir)
        print(f"Saving models to: {model_dir}")
        
        # Create checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.ppo_cfg["save_freq"],
            save_path=model_dir,
            name_prefix="ppo_model_step"
        )

        # Train the agent
        self.model.learn(
            total_timesteps=self.ppo_cfg["total_timesteps"],
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save final model
        final_path = os.path.join(model_dir, "final_model")
        self.save(final_path)
        print(f"Final model saved to: {final_path}")

    def predict(self, observation, deterministic=True):
        """Get action from the PPO model."""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path):
        """Save the PPO model."""
        self.model.save(path)

    def load(self, path):
        """Load the PPO model."""
        self.model = PPO.load(path)
        
    @staticmethod
    def get_latest_model_path(base_dir="saved_models"):
        """Find the latest saved model across all training runs."""
        # Check PPO directory
        ppo_dir = os.path.join(base_dir, "ppo")
        if not os.path.exists(ppo_dir):
            raise FileNotFoundError(f"No PPO models directory found at {ppo_dir}")
            
        # Get all training run directories
        run_dirs = [d for d in os.listdir(ppo_dir) 
                if os.path.isdir(os.path.join(ppo_dir, d))]
        if not run_dirs:
            raise FileNotFoundError(f"No training runs found in {ppo_dir}")
            
        # Get latest training run
        latest_run = sorted(run_dirs)[-1]
        run_dir = os.path.join(ppo_dir, latest_run)
        
        # Find latest model in that run
        models = [f for f in os.listdir(run_dir) 
                if f.startswith("ppo_model_step") or f == "final_model"]
        if not models:
            raise FileNotFoundError(f"No models found in {run_dir}")
            
        if "final_model" in models:
            latest_model = "final_model"
        else:
            # Extract step numbers and sort numerically
            def get_step_number(filename):
                try:
                    return int(filename.split('_')[-1])  # Extract the number at the end
                except ValueError:
                    return -1  # For files that don't match the pattern
                    
            models = sorted(models, key=get_step_number)
            latest_model = models[-1]
                
        model_path = os.path.join(run_dir, latest_model)
        return model_path