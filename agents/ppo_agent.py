from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import matplotlib.pyplot as plt
import numpy as np
import os
import gymnasium as gym

class MetricsCallback(BaseCallback):
    def __init__(self, agent, save_freq=1024, verbose=0):
        super().__init__(verbose)
        self.agent = agent
        self.save_freq = save_freq
        self.episode_step_count = 0
        self.current_episode_reward = 0
        self.speed_buffer = []

    def _on_step(self) -> bool:
        # Get info from current step
        infos = self.locals['infos']
        if len(infos) > 0:
            info = infos[0]
            
            # Update episode tracking
            self.episode_step_count += 1
            # rewards = reward = info.get('agents_rewards')
            # print(rewards)
            reward = info.get('agents_rewards', (0,))[0]
            print('reward', reward)
            self.current_episode_reward += reward
            
            # Track speed if available
            if hasattr(self.training_env.envs[0].unwrapped, 'controlled_vehicles'):
                vehicle = self.training_env.envs[0].unwrapped.controlled_vehicles[0]
                self.speed_buffer.append(vehicle.speed)

            # Get environment config for reward comparison
            env = self.training_env.envs[0].unwrapped
            
            # Check for collision and success
            # is_collision = reward == env.config["collision_reward"]
            # is_success = reward == env.config["arrived_reward"]
            is_collision = reward < 0
            is_success = reward ==5
            # Check if episode ended
            if self.locals['dones'][0]:
                # Record episode metrics
                self.agent.metrics['episode_rewards'].append(self.current_episode_reward)
                self.agent.metrics['episode_lengths'].append(self.episode_step_count)
                self.agent.metrics['collision_count'].append(1 if is_collision else 0)
                self.agent.metrics['success_count'].append(1 if is_success else 0)
                
                if len(self.speed_buffer) > 0:
                    self.agent.metrics['avg_speed'].append(np.mean(self.speed_buffer))
                else:
                    self.agent.metrics['avg_speed'].append(0)

                # Reset episode tracking
                self.episode_step_count = 0
                self.current_episode_reward = 0
                self.speed_buffer = []

        # Get training metrics
        if hasattr(self.model, 'logger'):
            logger = self.model.logger.name_to_value
            
            # Record training metrics if available
            if 'train/policy_gradient_loss' in logger:
                self.agent.metrics['policy_losses'].append(logger['train/policy_gradient_loss'])
                self.agent.metrics['value_losses'].append(logger['train/value_loss'])
                self.agent.metrics['entropy_losses'].append(logger['train/entropy_loss'])
                self.agent.metrics['approx_kl'].append(logger['train/approx_kl'])
                self.agent.metrics['explained_variance'].append(logger['train/explained_variance'])
                self.agent.metrics['clip_fraction'].append(logger['train/clip_fraction'])

        # Save metrics periodically
        if self.n_calls % self.save_freq == 0:
            self.agent.save_metrics()

        # Update plots occasionally
        if self.n_calls % 10 == 0:
            self.agent._update_plots()

        return True

class PPOAgent:
    def __init__(self, env, cfg):
        """
        Initialize PPO agent with metrics tracking.
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

        # Initialize metrics tracking
        self.metrics = {
            # Episode metrics
            'episode_rewards': [],
            'episode_lengths': [],
            'collision_count': [],
            'success_count': [],
            'avg_speed': [],
            
            # Training metrics
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'approx_kl': [],
            'explained_variance': [],
            'clip_fraction': [],
        }
        
        # Setup plotting
        self._setup_plots()

    def _setup_plots(self):
        """Initialize the plotting setup."""
        self.fig = plt.figure(figsize=(20, 15))
        gs = self.fig.add_gridspec(3, 2, hspace=0.3)
        
        # Episode Rewards subplot
        self.ax_rewards = self.fig.add_subplot(gs[0, 0])
        self.ax_rewards.set_title('Episode Rewards')
        self.ax_rewards.set_xlabel('Episode')
        self.ax_rewards.set_ylabel('Reward')
        self.reward_line, = self.ax_rewards.plot([], [], 'b-', label='Reward')
        self.ax_rewards.grid(True)
        self.ax_rewards.legend()

        # Success Rate subplot
        self.ax_success = self.fig.add_subplot(gs[0, 1])
        self.ax_success.set_title('Success & Collision Rate')
        self.ax_success.set_xlabel('Episode')
        self.ax_success.set_ylabel('Rate')
        self.success_line, = self.ax_success.plot([], [], 'g-', label='Success Rate')
        self.collision_line, = self.ax_success.plot([], [], 'r-', label='Collision Rate')
        self.ax_success.grid(True)
        self.ax_success.legend()

        # Training Losses subplot
        self.ax_losses = self.fig.add_subplot(gs[1, 0])
        self.ax_losses.set_title('Training Losses')
        self.ax_losses.set_xlabel('Update')
        self.ax_losses.set_ylabel('Loss Value')
        self.policy_loss_line, = self.ax_losses.plot([], [], 'r-', label='Policy Loss')
        self.value_loss_line, = self.ax_losses.plot([], [], 'b-', label='Value Loss')
        self.entropy_loss_line, = self.ax_losses.plot([], [], 'g-', label='Entropy Loss')
        self.ax_losses.grid(True)
        self.ax_losses.legend()
        self.ax_losses.set_yscale('log')

        # KL & Clip Fraction subplot
        self.ax_kl = self.fig.add_subplot(gs[1, 1])
        self.ax_kl.set_title('KL Divergence & Clip Fraction')
        self.ax_kl.set_xlabel('Update')
        self.kl_line, = self.ax_kl.plot([], [], 'b-', label='Approx KL')
        self.clip_line, = self.ax_kl.plot([], [], 'r-', label='Clip Fraction')
        self.ax_kl.grid(True)
        self.ax_kl.legend()

        # Episode Length subplot
        self.ax_length = self.fig.add_subplot(gs[2, 0])
        self.ax_length.set_title('Episode Length')
        self.ax_length.set_xlabel('Episode')
        self.ax_length.set_ylabel('Steps')
        self.length_line, = self.ax_length.plot([], [], 'b-', label='Length')
        self.ax_length.grid(True)
        self.ax_length.legend()

        # Average Speed subplot
        self.ax_speed = self.fig.add_subplot(gs[2, 1])
        self.ax_speed.set_title('Average Speed')
        self.ax_speed.set_xlabel('Episode')
        self.ax_speed.set_ylabel('Speed')
        self.speed_line, = self.ax_speed.plot([], [], 'b-', label='Avg Speed')
        self.ax_speed.grid(True)
        self.ax_speed.legend()

        plt.tight_layout()
        plt.ion()
        plt.show(block=False)

    def _update_plots(self):
        """Update all plots with current metrics."""
        # Update episode rewards
        if len(self.metrics['episode_rewards']) > 0:
            self.reward_line.set_data(range(len(self.metrics['episode_rewards'])), 
                                    self.metrics['episode_rewards'])
            self.ax_rewards.relim()
            self.ax_rewards.autoscale_view()

        # Update success and collision rates
        if len(self.metrics['success_count']) > 0:
            episodes = range(len(self.metrics['success_count']))
            success_rate = [sum(self.metrics['success_count'][:i+1])/(i+1) 
                          for i in range(len(self.metrics['success_count']))]
            collision_rate = [sum(self.metrics['collision_count'][:i+1])/(i+1) 
                            for i in range(len(self.metrics['collision_count']))]
            
            self.success_line.set_data(episodes, success_rate)
            self.collision_line.set_data(episodes, collision_rate)
            self.ax_success.relim()
            self.ax_success.autoscale_view()

        # Update losses
        if len(self.metrics['policy_losses']) > 0:
            updates = range(len(self.metrics['policy_losses']))
            self.policy_loss_line.set_data(updates, self.metrics['policy_losses'])
            self.value_loss_line.set_data(updates, self.metrics['value_losses'])
            self.entropy_loss_line.set_data(updates, self.metrics['entropy_losses'])
            self.ax_losses.relim()
            self.ax_losses.autoscale_view()

        # Update KL and clip fraction
        if len(self.metrics['approx_kl']) > 0:
            updates = range(len(self.metrics['approx_kl']))
            self.kl_line.set_data(updates, self.metrics['approx_kl'])
            self.clip_line.set_data(updates, self.metrics['clip_fraction'])
            self.ax_kl.relim()
            self.ax_kl.autoscale_view()

        # Update episode length
        if len(self.metrics['episode_lengths']) > 0:
            self.length_line.set_data(range(len(self.metrics['episode_lengths'])), 
                                    self.metrics['episode_lengths'])
            self.ax_length.relim()
            self.ax_length.autoscale_view()

        # Update average speed
        if len(self.metrics['avg_speed']) > 0:
            self.speed_line.set_data(range(len(self.metrics['avg_speed'])), 
                                   self.metrics['avg_speed'])
            self.ax_speed.relim()
            self.ax_speed.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save_metrics(self, path="./saved_metrics/ppo"):
        """Save metrics to file."""
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/training_metrics.npy", self.metrics)
        self.fig.savefig(f"{path}/training_plots.png")

    def load_metrics(self, path="./saved_metrics/ppo"):
        """Load previously saved metrics."""
        if os.path.exists(f"{path}/training_metrics.npy"):
            self.metrics = np.load(f"{path}/training_metrics.npy", allow_pickle=True).item()
            self._update_plots()

    def learn(self, save_dir="saved_models"):
        """Train the PPO agent with metrics tracking."""
        # Setup save directory
        model_dir = self._setup_save_directory(save_dir)
        print(f"Saving models to: {model_dir}")
        
        # Try to load previous metrics if they exist
        self.load_metrics()

        # Create callbacks
        metrics_callback = MetricsCallback(self, save_freq=1024)
        checkpoint_callback = CheckpointCallback(
            save_freq=self.ppo_cfg["save_freq"],
            save_path=model_dir,
            name_prefix="ppo_model_step"
        )

        # Train the agent
        self.model.learn(
            total_timesteps=self.ppo_cfg["total_timesteps"],
            callback=[metrics_callback, checkpoint_callback],
            progress_bar=True
        )
        
        # Save final model and metrics
        final_path = os.path.join(model_dir, "final_model")
        self.save(final_path)
        self.save_metrics()
        print(f"Final model saved to: {final_path}")
        
        # Show final plots
        plt.show(block=True)

    def _setup_save_directory(self, base_dir="saved_models"):
        """Create a new subfolder for this training run."""
        import datetime
        ppo_dir = os.path.join(base_dir, "ppo")
        os.makedirs(ppo_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(ppo_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)
        
        return save_dir

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