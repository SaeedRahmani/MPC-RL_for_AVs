import hydra
import numpy as np
import gymnasium as gym
from typing import Dict

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
        # "sac": SAC,
        # "td3": TD3,
        # "ddpg": DDPG,
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

        # Build the policy (neural network) with modified action_dim
        self.algo: BaseAlgorithm = self._specify_algo()

        # Create the model with a custom policy
        self.model = PPO_MPC(
            policy=create_a2c_policy(self.mpcrl_cfg["action_space_dim"]),
            env=self.env,
            pure_mpc_cfg=self.pure_mpc_cfg
        )

        import numpy as np
        from gymnasium.spaces import Box

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
    def __init__(self, cfg):
        super(RefSpeedTrainer, self).__init__()


class DynamicWeightTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(DynamicWeightTrainer, self).__init__()


@hydra.main(config_name="cfg", config_path="../config", version_base="1.3")
def test_trainer(cfg):

    import highway_env
    
    gym_env_config = build_env_config(cfg)
    mpcrl_agent_config = build_mpcrl_agent_config(cfg)
    pure_mpc_agent_config = build_pure_mpc_agent_config(cfg)

    # env
    env = gym.make("intersection-v1", render_mode="rgb_array", config=gym_env_config)

    trainer = BaseTrainer(env, mpcrl_agent_config, pure_mpc_agent_config)

    # print(trainer.model.policy)

    trainer.model.learn(
        total_timesteps=mpcrl_agent_config["total_timesteps"], 
        # callback=create_callback_func(),
        progress_bar=mpcrl_agent_config["show_progress_bar"])

if __name__ == "__main__":
    test_trainer()