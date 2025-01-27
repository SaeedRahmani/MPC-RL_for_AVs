import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from typing import Dict

class TrainingMonitorCallback(BaseCallback):
    """
    Custom callback for monitoring training progress
    """
    def __init__(self, check_freq: int = 1000, log_dir: str = None, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        
        # History tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[float] = []
        self.training_metrics: Dict[str, List[float]] = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'learning_rate': [],
            'explained_variance': []
        }
        
        # Performance tracking
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        
        # Initialize plots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        plt.ion()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get episode statistics
            episode_rewards = self.model.ep_info_buffer
            if len(episode_rewards) > 0:
                mean_reward = np.mean([ep['r'] for ep in episode_rewards])
                mean_length = np.mean([ep['l'] for ep in episode_rewards])
                
                self.episode_rewards.append(mean_reward)
                self.episode_lengths.append(mean_length)
                
                # Get training metrics
                if hasattr(self.model, 'logger'):
                    for key in self.training_metrics.keys():
                        if key in self.model.logger.name_to_value:
                            self.training_metrics[key].append(
                                self.model.logger.name_to_value[key]
                            )
                
                # Update plots
                self._update_plots()
                
                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.log_dir is not None:
                        self.model.save(f"{self.log_dir}/best_model")
                
                # Log improvement
                if self.verbose > 0:
                    print(f"Timesteps: {self.num_timesteps}")
                    print(f"Mean reward: {mean_reward:.2f}")
                    print(f"Mean episode length: {mean_length:.2f}")
                    if self.last_mean_reward > -np.inf:
                        improvement = mean_reward - self.last_mean_reward
                        print(f"Improvement: {improvement:.2f}")
                    self.last_mean_reward = mean_reward

        return True

    def _update_plots(self):
        """Update training visualization plots"""
        # Clear all subplots
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot episode rewards
        self.axes[0, 0].plot(self.episode_rewards)
        self.axes[0, 0].set_title('Episode Rewards')
        self.axes[0, 0].set_xlabel('Episode')
        self.axes[0, 0].set_ylabel('Reward')
        self.axes[0, 0].grid(True)
        
        # Plot episode lengths
        self.axes[0, 1].plot(self.episode_lengths)
        self.axes[0, 1].set_title('Episode Lengths')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].set_ylabel('Length')
        self.axes[0, 1].grid(True)
        
        # Plot training losses
        if len(self.training_metrics['policy_loss']) > 0:
            self.axes[1, 0].plot(
                self.training_metrics['policy_loss'],
                label='Policy Loss'
            )
            self.axes[1, 0].plot(
                self.training_metrics['value_loss'],
                label='Value Loss'
            )
            self.axes[1, 0].set_title('Training Losses')
            self.axes[1, 0].set_xlabel('Update')
            self.axes[1, 0].set_ylabel('Loss')
            self.axes[1, 0].legend()
            self.axes[1, 0].grid(True)
        
        # Plot other metrics
        if len(self.training_metrics['explained_variance']) > 0:
            self.axes[1, 1].plot(
                self.training_metrics['explained_variance'],
                label='Explained Variance'
            )
            self.axes[1, 1].plot(
                self.training_metrics['entropy_loss'],
                label='Entropy Loss'
            )
            self.axes[1, 1].set_title('Other Metrics')
            self.axes[1, 1].set_xlabel('Update')
            self.axes[1, 1].set_ylabel('Value')
            self.axes[1, 1].legend()
            self.axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)