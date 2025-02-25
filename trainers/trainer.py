import matplotlib.pyplot as plt 
import hydra
import numpy as np
import torch
import gymnasium as gym
from typing import Dict
        
from gymnasium.spaces import Box
from stable_baselines3 import A2C, PPO       # Off-policy
from stable_baselines3 import SAC, TD3, DDPG # On-policy
from stable_baselines3 import DQN

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback,  CallbackList

from config.config import build_env_config, build_mpcrl_agent_config, build_pure_mpc_agent_config
from trainers.trainer_utils import create_a2c_policy, create_ppo_policy, create_callback_func
from agents.a2c_mpc import A2C_MPC
from agents.ppo_mpc import PPO_MPC

import os 

class BaseTrainer:
    """ A base trainer class for reinforcement learning algorithms. """

    ALGO: Dict[str, BaseAlgorithm] = {
        "ppo": PPO_MPC,
        "a2c": A2C_MPC,
    }

    def __init__(self, env: gym.Env, mpcrl_cfg: dict, pure_mpc_cfg: dict):
        
        """
        Initialize the BaseTrainer.
        Args:
            env: The gym environment
            mpcrl_cfg: MPC-RL configuration dict
            pure_mpc_cfg: Pure MPC configuration dict
            enable_viz: Whether to enable visualization during training
        """
        plt.close('all')
        

        self.env = env
        self.mpcrl_cfg = mpcrl_cfg
        self.pure_mpc_cfg = pure_mpc_cfg
        self.model = None
        
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
        
        # Create and setup figure with 3x2 subplots
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

    def save_metrics(self, path="./saved_metrics"):
        """Save the current metrics to a file."""
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/training_metrics.npy", self.metrics)
        self.fig.savefig(f"{path}/training_plots.png")

    def load_metrics(self, path="./saved_metrics"):
        """Load previously saved metrics."""
        if os.path.exists(f"{path}/training_metrics.npy"):
            self.metrics = np.load(f"{path}/training_metrics.npy", allow_pickle=True).item()
            self._update_plots()

    def learn(self):
        self._build_model()

        # Initialize the save callback
        save_callback = SaveModelCallback(
            save_path="./saved_models",
            save_freq=256,
            verbose=1
        )

        # Define custom callback for plotting
        class PlottingCallback(BaseCallback):
            def __init__(self, trainer, verbose=0):
                super().__init__(verbose)
                self.trainer = trainer

            def _on_step(self):
                # Get latest reward and update step rewards
                if len(self.locals['infos']) > 0:
                    latest_reward = self.locals['infos'][0].get('agents_rewards', (0,))[0]
                    self.trainer.step_rewards.append(latest_reward)
                    self.trainer.current_episode_rewards.append(latest_reward)

                    # Check if episode has ended
                    if self.locals['dones'][0]:
                        if len(self.trainer.current_episode_rewards) > 0:
                            episode_avg_reward = np.mean(self.trainer.current_episode_rewards)
                            self.trainer.episode_rewards.append(episode_avg_reward)
                            self.trainer.current_episode_rewards = []  # Reset for next episode

                # Get latest loss
                if hasattr(self.model, 'logger'):
                    logger_values = self.model.logger.name_to_value
                    if len(logger_values) > 0:
                        loss = None
                        for key in logger_values:
                            if 'loss' in key.lower():
                                loss = logger_values[key]
                                break
                        if loss is not None:
                            self.trainer.losses.append(loss)

                # Update plots every 10 steps
                if self.n_calls % 10 == 0:
                    self.trainer._update_plots()
                return True

        plotting_callback = PlottingCallback(self)

        # Combine both callbacks
        combined_callbacks = CallbackList([save_callback, plotting_callback])

        self.model.learn(
            total_timesteps=self.mpcrl_cfg["total_timesteps"],
            progress_bar=self.mpcrl_cfg["show_progress_bar"],
            callback=combined_callbacks,
        )

        # Final plot update and display
        self._update_plots()
        plt.figure(self.fig.number)
        plt.show(block=True)
    def learn(self):
        self._build_model()

        # Try to load previous metrics if they exist
        self.load_metrics()

        # Create callbacks
        metrics_callback = MetricsCallback(self, save_freq=1024)
        save_callback = SaveModelCallback(
            save_path="./saved_models",
            save_freq=1024,
            verbose=1
        )

        # Combine callbacks
        combined_callbacks = CallbackList([metrics_callback, save_callback])

        self.model.learn(
            total_timesteps=self.mpcrl_cfg["total_timesteps"],
            progress_bar=self.mpcrl_cfg["show_progress_bar"],
            callback=combined_callbacks,
        )

        # Final metrics save and plot update
        self.save_metrics()
        self._update_plots()
        plt.show(block=True)

    def learn_old(self):
        self._build_model()

        # Initialize the save callback
        save_callback = SaveModelCallback(
            save_path="./saved_models/mpcrl",
            save_freq=1024,
            verbose=1
        )

        # Define the plotting and logging callback
        def plotting_callback(_locals, _globals):
            if _locals['dones']:
                self.rewards.append(_locals['rewards'][0])
            
            if hasattr(_locals['self'], 'logger'):
                if len(_locals['self'].logger.name_to_value) > 0:
                    loss = _locals['self'].logger.name_to_value.get('train/loss', 0)
                    if loss > 0:
                        self.losses.append(loss)
                    
            if len(self.rewards) % 10 == 0:
                self._update_plots()
            return True

        # Combine both callbacks into a CallbackList
        combined_callback = CallbackList([save_callback])

        self.model.learn(
            total_timesteps=self.mpcrl_cfg["total_timesteps"],
            progress_bar=self.mpcrl_cfg["show_progress_bar"],
            callback=combined_callback,
        )

        # Final plot update and display
        self._update_plots()
        plt.figure(self.fig.number)  # Ensure correct figure is active
        plt.show(block=True)  # Keep plots open after training



    # Old learn method before adding the plotting
    # def learn(self):
    #     self.model.learn(
    #         total_timesteps=self.mpcrl_cfg["total_timesteps"], 
    #         progress_bar=self.mpcrl_cfg["show_progress_bar"],
    #     )

    def save(self, file, path):
        self.model.save(f"./{path}/{self.version}/{file}")
        # Save figure without closing
        plt.figure(self.fig.number)
        plt.savefig(f"{path}/training_plots_{file}.png")
        plt.show(block=True)  # Keep displaying after save

    def load(self, path, mpcrl_cfg, version, pure_mpc_cfg, env):
        self.model.load(path, mpcrl_cfg, version, pure_mpc_cfg, env)

    def predict(self, obs, return_numpy = True):
        self.model.policy.eval()
        
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.Tensor(obs).flatten().unsqueeze(dim=0)
        RL_output, _, _ = self.model.policy(obs_tensor)


        if self.model.version == "v0":
            mpc_action = self.model.mpc_agent.predict(
                obs=obs,
                return_numpy=return_numpy,
                weights_from_RL=None,
                ref_speed=RL_output.detach().numpy(),
            )
            # Print the reference speed
            print("Reference Speed from RL Agent:", RL_output.detach().numpy())
        else:
            mpc_action = self.model.mpc_agent.predict(
                obs=obs,
                return_numpy=return_numpy,
                weights_from_RL=RL_output.detach().numpy(),
                ref_speed=None,
            )
            # Print the reference speed
            print("Reference Speed from RL Agent:", RL_output.detach().numpy())
            # mpc_action = mpc_action.reshape((1,2))    
        return mpc_action

    def _setup_algorithm(self):
        algorithm = self.mpcrl_cfg["algorithm"]
        self.algo = PPO_MPC if algorithm == "ppo" else A2C_MPC

  
    def _build_model(self):
        # Determine algorithm and build policy
        self.algo: BaseAlgorithm = self._specify_algo()
        algorithm = self.mpcrl_cfg["algorithm"]
        action_dim = self.mpcrl_cfg["action_space_dim"]

        # Select appropriate policy creation function
        if algorithm == "a2c":
            policy = create_a2c_policy(action_dim)
        elif algorithm == "ppo":
            policy = create_ppo_policy(action_dim)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Common parameters for both algorithms
        common_params = {
            "policy": policy,
            "env": self.env,
            "mpcrl_cfg": self.mpcrl_cfg,
            "version": self.version,
            "pure_mpc_cfg": self.pure_mpc_cfg,
            "learning_rate": self.mpcrl_cfg["learning_rate"],
            "n_steps": self.mpcrl_cfg["n_steps"],
            "gamma": self.mpcrl_cfg.get("gamma", 0.99),
            "gae_lambda": self.mpcrl_cfg.get("gae_lambda", 0.95),
            "ent_coef": self.mpcrl_cfg.get("ent_coef", 0.0),
            "vf_coef": self.mpcrl_cfg.get("vf_coef", 0.5),
            "max_grad_norm": self.mpcrl_cfg.get("max_grad_norm", 0.5),
        }

        # Algorithm-specific parameters
        if algorithm == "ppo":
            common_params.update({
                "batch_size": self.mpcrl_cfg["batch_size"],
                "n_epochs": self.mpcrl_cfg["n_epochs"],
                "clip_range": self.mpcrl_cfg.get("clip_range", 0.2),
                "clip_range_vf": self.mpcrl_cfg.get("clip_range_vf", None),
            })
        elif algorithm == "a2c":
            common_params.update({
                "rms_prop_eps": self.mpcrl_cfg.get("rms_prop_eps", 1e-5),
                "use_rms_prop": self.mpcrl_cfg.get("use_rms_prop", True),
            })

        # Initialize the model with the selected algorithm and parameters
        self.model = self.algo(**common_params)

        # TODO: Check to see if this is generic 
        # Replace the action space in the model
        new_action_space = Box(
            low=-1 * np.ones(action_dim),
            high=np.ones(action_dim),
            shape=(action_dim,),
            dtype=np.float32,
        )
        self.model.action_space = new_action_space
        
        # if self.version == "v0":
        #     low = 0.0   # Lower bound for v0
        #     high = 30.0  # Upper bound for v0
        # elif self.version == "v1":
        #     low = 0.0        # Lower bounds for 6 weights in v1
        #     high = 300.0 # Upper bounds for 6 weights in v1
        # else:
        #     raise ValueError("Unknown version: must be 'v0' or 'v1'")

        # # Define the action space
        # new_action_space = Box(
        #     low=low, 
        #     high=high, 
        #     shape=(action_dim,), 
        #     dtype=np.float32)
        
        
        self.action_space = new_action_space  # Assign to the model's action space

        # Replace the action dimension in the rollout buffer, if applicable
        if hasattr(self.model, "rollout_buffer") and self.model.rollout_buffer is not None:
            self.model.rollout_buffer.action_dim = action_dim
        
    def _specify_algo(self) -> BaseAlgorithm:
        """
        Specify the family of RL algorithm to use.

        :return: An instance of the specified reinforcement learning algorithm.
        """
        algorithm = self.mpcrl_cfg["algorithm"]  # Changed from "algo" to "algorithm"
        assert algorithm in BaseTrainer.ALGO, f"Algorithm '{algorithm}' is not supported."
        return BaseTrainer.ALGO[algorithm]

class MetricsCallback(BaseCallback):
    def __init__(self, trainer, save_freq=1024, verbose=0):
        super().__init__(verbose)
        self.trainer = trainer
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
            reward = info.get('agents_rewards', (0,))[0]  # Instant reward
            self.current_episode_reward += reward
            
            # Track speed
            if hasattr(self.trainer.env.unwrapped, 'controlled_vehicles'):
                vehicle = self.trainer.env.unwrapped.controlled_vehicles[0]
                self.speed_buffer.append(vehicle.speed)

            # Print rewards for debugging
            env = self.trainer.env.unwrapped
            print(f"Step Reward: {reward}, Collision Reward: {env.config['collision_reward']}, Arrival Reward: {env.config['arrived_reward']}")

            # Check for collision and success
            # is_collision = abs(reward - env.config["collision_reward"]) < 1e-5
            is_collision = reward == env.config["collision_reward"]
            is_success = reward == env.config["arrived_reward"]

            # Check if episode ended
            if self.locals['dones'][0]:  
                # Record episode metrics
                self.trainer.metrics['episode_rewards'].append(self.current_episode_reward)
                self.trainer.metrics['episode_lengths'].append(self.episode_step_count)
                self.trainer.metrics['collision_count'].append(1 if is_collision else 0)
                self.trainer.metrics['success_count'].append(1 if is_success else 0)
                
                if len(self.speed_buffer) > 0:
                    self.trainer.metrics['avg_speed'].append(np.mean(self.speed_buffer))
                else:
                    self.trainer.metrics['avg_speed'].append(0)

                # Reset episode tracking
                self.episode_step_count = 0
                self.current_episode_reward = 0
                self.speed_buffer = []

        # Get training metrics
        if hasattr(self.model, 'logger'):
            logger = self.model.logger.name_to_value
            
            # Record training metrics if available
            if 'train/policy_gradient_loss' in logger:
                self.trainer.metrics['policy_losses'].append(logger['train/policy_gradient_loss'])
                self.trainer.metrics['value_losses'].append(logger['train/value_loss'])
                self.trainer.metrics['entropy_losses'].append(logger['train/entropy_loss'])
                self.trainer.metrics['approx_kl'].append(logger['train/approx_kl'])
                self.trainer.metrics['explained_variance'].append(logger['train/explained_variance'])
                self.trainer.metrics['clip_fraction'].append(logger['train/clip_fraction'])

        # Save metrics periodically
        if self.n_calls % self.save_freq == 0:
            self.trainer.save_metrics()

        # Update plots occasionally
        if self.n_calls % 10 == 0:
            self.trainer._update_plots()

        return True



class RefSpeedTrainer(BaseTrainer):
    def __init__(self, env: gym.Env, mpcrl_cfg: dict, pure_mpc_cfg: dict):
        super(RefSpeedTrainer, self).__init__(env, mpcrl_cfg, pure_mpc_cfg)
        self.version = "v0"
        self._setup_algorithm()
        self._build_model()
    # def _build_model(self, version="v0"):
    #     return super()._build_model(version)

class DynamicWeightTrainer(BaseTrainer):
    def __init__(self, env: gym.Env, mpcrl_cfg: dict, pure_mpc_cfg: dict):
        super(DynamicWeightTrainer, self).__init__(env, mpcrl_cfg, pure_mpc_cfg)
        self.version = "v1"
        self._setup_algorithm()
        self._build_model()
    # def _build_model(self, version="v1"):
    #     return super()._build_model(version)

class SaveModelCallback(BaseCallback):
    def __init__(self, save_path, save_freq, verbose=0):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            save_file = f"{self.save_path}/model_step_{self.n_calls}"
            self.model.save(save_file)
            if self.verbose > 0:
                print(f"Model saved to {save_file}")
        return True

@hydra.main(config_name="cfg", config_path="../config", version_base="1.3")
def test_trainer(cfg):

    import highway_env
    
    gym_env_config = build_env_config(cfg)
    mpcrl_agent_config = build_mpcrl_agent_config(cfg)
    pure_mpc_agent_config = build_pure_mpc_agent_config(cfg)

    # env
    env = gym.make("intersection-v1", render_mode="rgb_array", config=gym_env_config)

    # trainer = DynamicWeightTrainer(env, mpcrl_agent_config, pure_mpc_agent_config)
    # trainer.learn()
    # trainer.save()

    trainer = RefSpeedTrainer(env, mpcrl_agent_config, pure_mpc_agent_config, env)
    trainer.learn()

if __name__ == "__main__":
    test_trainer()