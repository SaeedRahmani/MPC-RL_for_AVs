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


class BaseTrainer:
    """ A base trainer class for reinforcement learning algorithms. """

    ALGO: Dict[str, BaseAlgorithm] = {
        "ppo": PPO_MPC,
        "a2c": A2C_MPC,
    }

    def __init__(self, env: gym.Env, mpcrl_cfg: dict, pure_mpc_cfg: dict):
        """
        Initialize the BaseTrainer.

        :param env: The gym environment.
        :param cfg: Configuration dictionary containing algorithm type and action space dimensions.
        """
        plt.close('all')

        self.env = env
        self.mpcrl_cfg = mpcrl_cfg
        self.pure_mpc_cfg = pure_mpc_cfg
        self.model = None

        # moved to RefSpeedTrainer and DynamicWeightTrainer
        # self._setup_algorithm()
        # self._build_model()
        
        
        # Initialize plotting data
        self.rewards = []
        self.losses = []
        
        # Create and setup figure
        self.fig = plt.figure(figsize=(12, 8))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
        
        # Setup reward subplot
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax1.set_title('Episode Rewards', fontsize=12, pad=10)
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Reward Value')
        self.reward_line, = self.ax1.plot([], [], 'b-', label='Reward')
        self.ax1.grid(True)
        self.ax1.legend()

        # Setup loss subplot
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax2.set_title('Training Loss', fontsize=12, pad=10)
        self.ax2.set_xlabel('Training Step')
        self.ax2.set_ylabel('Loss Value (log scale)')
        self.ax2.set_yscale('log')
        self.loss_line, = self.ax2.plot([], [], 'r-', label='Loss')
        self.ax2.grid(True)
        self.ax2.legend()

        # Add main title
        self.fig.suptitle('Training Progress', fontsize=14, y=0.95)
        
        # Show the figure
        plt.ion()  # Enable interactive mode
        plt.tight_layout()
        plt.show(block=False)

    def _update_plots(self):
        if len(self.rewards) > 0:
            self.reward_line.set_data(range(len(self.rewards)), self.rewards)
            self.ax1.relim()
            self.ax1.autoscale_view()
            
        if len(self.losses) > 0:
            self.loss_line.set_data(range(len(self.losses)), self.losses)
            self.ax2.relim()
            self.ax2.autoscale_view()
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def learn(self):
        self._build_model()

        # Initialize the save callback
        save_callback = SaveModelCallback(
            save_path="./saved_models",
            save_freq=1024,
            verbose=1
        )

        # Define custom callback for plotting
        class PlottingCallback(BaseCallback):
            def __init__(self, trainer, verbose=0):
                super().__init__(verbose)
                self.trainer = trainer

            def _on_step(self):
                # Get latest reward
                if len(self.locals['infos']) > 0:
                    latest_reward = self.locals['infos'][0].get('agents_rewards', (0,))[0]
                    self.trainer.rewards.append(latest_reward)

                # Get latest loss - print all available keys for debugging
                if hasattr(self.model, 'logger'):
                    logger_values = self.model.logger.name_to_value
                    if len(logger_values) > 0:
                        print("Available logger keys:", logger_values.keys())
                        # Try to get any loss-related value
                        loss = None
                        for key in logger_values:
                            if 'loss' in key.lower():
                                loss = logger_values[key]
                                break
                        if loss is not None:
                            print(f"Captured loss value: {loss}")
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

    def learn_old(self):
        self._build_model()

        # Initialize the save callback
        save_callback = SaveModelCallback(
            save_path="./saved_models",
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

        # Replace the action space in the model
        new_action_space = Box(
            low=-1 * np.ones(action_dim),
            high=np.ones(action_dim),
            shape=(action_dim,),
            dtype=np.float32,
        )
        self.model.action_space = new_action_space

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