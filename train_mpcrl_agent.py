import gymnasium as gym
import highway_env
import hydra
import numpy as np

from agents.mpcrl_agent import MPCRL_Agent
from config.config import build_env_config, build_pure_mpc_agent_config
from stable_baselines3 import PPO

@hydra.main(config_name="cfg", config_path="./config", version_base="1.3")
def train_mpcrl_agent(cfg):
    # config
    gym_env_config = build_env_config(cfg)
    pure_mpc_agent_config = build_pure_mpc_agent_config(cfg)

    # 创建环境实例
    env = gym.make("intersection-v1", render_mode="rgb_array", config=gym_env_config)

    # 创建 PPO 模型
    model = PPO("MlpPolicy", env, verbose=1)

    # 训练参数
    total_timesteps = 10000
    update_frequency = 1000  # 每多少步更新模型

    # 初始化
    obs, info = env.reset()
    for step in range(total_timesteps):
        # 获取动作和网络输出
        action, _states = model.predict(obs, deterministic=True)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 进行模型学习
        model.replay_buffer.add(obs, action, reward, terminated)  # 将当前经验添加到重放缓冲区
        model.learn(total_timesteps=update_frequency)  # 定期更新模型
        
        # 打印网络输出（例如动作概率或 Q 值）
        if hasattr(model.policy, "logits"):
            logits = model.policy.logits(obs)  # 获取网络输出
            print("Step:", step, "Logits:", logits.numpy())
        
        if done:
            obs = env.reset()  # 如果完成，重置环境

    # 最后更新模型一次
    model.learn(total_timesteps=1)

if __name__ == "__main__":
    train_mpcrl_agent()