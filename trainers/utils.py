import numpy as np
from gymnasium.spaces import Box
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback

def create_a2c_policy(action_dim: int):
    """
    Dynamically return an A2C Policy class with a specified action dimension.

    :param action_dim: The dimension of the action space.
    :return: A class inheriting from ActorCriticPolicy with a customized action space.
    """

    class A2C_Policy(ActorCriticPolicy):
        def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, **kwargs):
            # Set the action space based on the provided action_dim
            action_space = Box(
                low=-1 * np.ones(action_dim),
                high=np.ones(action_dim),
                shape=(action_dim,),
                dtype=np.float32,
            )

            super().__init__(observation_space, action_space, lr_schedule, net_arch, **kwargs)

    return A2C_Policy  # Return the dynamically created class

def create_ppo_policy(action_dim: int):
    """
    Dynamically return a PPO Policy class with a specified action dimension.
    """
    class PPO_Policy(ActorCriticPolicy):
        def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, **kwargs):
            # Set the action space based on the provided action_dim
            action_space = Box(
                low=-1 * np.ones(action_dim),
                high=np.ones(action_dim),
                shape=(action_dim,),
                dtype=np.float32,
            )

            super().__init__(observation_space, action_space, lr_schedule, net_arch, **kwargs)

    return PPO_Policy  # Return the dynamically created class

def create_callback_func():
    """
    Dynamically create and return an MPC Callback class.

    This callback process the action generated by RL agent and let MPC agent to process.
    """
    class MPC_Callback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
    
        def _on_step(self) -> bool:
            # Print available keys and actions for debugging purposes
            # action_info = self.locals.get("actions")
            # print(f"Available locals: {self.locals.keys()}, Actions: {action_info.shape}")
            self.locals["actions"] = np.zeros((1,2,3))
            return True  # Continue training

    return MPC_Callback()  # Return the dynamically created callback class