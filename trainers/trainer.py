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
from stable_baselines3.common.callbacks import BaseCallback

from config.config import build_env_config, build_mpcrl_agent_config, build_pure_mpc_agent_config
from trainers.utils import create_a2c_policy, create_callback_func
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
        self.mpcrl_cfg = mpcrl_cfg
        self.pure_mpc_cfg = pure_mpc_cfg
        self.env: gym.Env = env

        self._build_model()

    def learn(self):
        self.model.learn(
            total_timesteps=self.mpcrl_cfg["total_timesteps"], 
            progress_bar=self.mpcrl_cfg["show_progress_bar"],
        )

    def save(self, file, path):
        self.model.save(f"{path}/{self.version}/{file}")    

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
        else:
            mpc_action = self.model.mpc_agent.predict(
                obs=obs,
                return_numpy=return_numpy,
                weights_from_RL=RL_output.detach().numpy(),
                ref_speed=None,
            )            
            # mpc_action = mpc_action.reshape((1,2))    
        return mpc_action

    def _build_model(self, version="v1"):
        # Build the policy (neural network) with modified action_dim
        self.algo: BaseAlgorithm = self._specify_algo()

        # Create the model with a custom policy
        self.model = self.algo(
            policy=create_a2c_policy(self.mpcrl_cfg["action_space_dim"]),
            env=self.env,
            mpcrl_cfg=self.mpcrl_cfg,
            version=version,
            pure_mpc_cfg=self.pure_mpc_cfg,
            learning_rate=self.mpcrl_cfg["action_space_dim"],
            n_steps=self.mpcrl_cfg["n_steps"],
            batch_size=self.mpcrl_cfg["batch_size"],
            n_epochs=self.mpcrl_cfg["n_epochs"],
        )

        # replace the action_space
        self.model.action_space = Box(
                low=-1 * np.ones(self.mpcrl_cfg["action_space_dim"]),
                high=np.ones(self.mpcrl_cfg["action_space_dim"]),
                shape=(self.mpcrl_cfg["action_space_dim"],),
                dtype=np.float32,
            )
        self.model.rollout_buffer.action_dim = self.mpcrl_cfg["action_space_dim"]
        
    def _specify_algo(self) -> BaseAlgorithm:
        """
        Specify the family of RL algorithm to use.

        :return: An instance of the specified reinforcement learning algorithm.
        """
        algo_name = self.mpcrl_cfg["algo"]
        assert algo_name in BaseTrainer.ALGO, f"Algorithm '{algo_name}' is not supported."
        return BaseTrainer.ALGO[algo_name]


class RefSpeedTrainer(BaseTrainer):
    def __init__(self, env: gym.Env, mpcrl_cfg: dict, pure_mpc_cfg: dict):
        super(RefSpeedTrainer, self).__init__(env, mpcrl_cfg, pure_mpc_cfg)

    def _build_model(self, version="v0"):
        return super()._build_model(version)

class DynamicWeightTrainer(BaseTrainer):
    def __init__(self, env: gym.Env, mpcrl_cfg: dict, pure_mpc_cfg: dict):
        super(DynamicWeightTrainer, self).__init__(env, mpcrl_cfg, pure_mpc_cfg)

    def _build_model(self, version="v1"):
        return super()._build_model(version)

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