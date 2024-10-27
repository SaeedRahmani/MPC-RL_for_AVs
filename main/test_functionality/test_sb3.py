import torch
import torch.nn as nn
import gymnasium
import numpy as np
import highway_env
from gymnasium.spaces import Box
from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, **kwargs):
        # 在这里定义自定义的动作空间维度
        self.action_dim = 300  # 自定义动作空间的维度
        self.action_space = Box(
            high=np.ones((self.action_dim)),
            low=np.ones((self.action_dim)) * -1,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        # 使用默认的网络架构初始化
        super(CustomPolicy, self).__init__(observation_space, self.action_space, lr_schedule, net_arch, **kwargs)

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        print(self.locals.keys(), self.locals["actions"])
        self.locals["actions"] = np.array(([0,0]))
        return False  # 继续训练


env = gymnasium.make("intersection-v1")
print(env.action_space)
# 使用自定义策略创建 A2C 模型
model = A2C(CustomPolicy, env, verbose=1)

# model.learn(int(2e4), progress_bar=False, callback=CustomCallback())
print(model.policy)