import gymnasium as gym
import highway_env
import hydra
import numpy as np

from agents.pure_mpc_linear import IterativeLinearMPC_Agent
from config.config import build_env_config, build_pure_mpc_agent_config

np.set_printoptions(suppress=True)

@hydra.main(config_name="cfg", config_path="./config", version_base="1.3")
def test_iterative_linear_mpc_agent(cfg):
    # config
    gym_env_config = build_env_config(cfg)
    pure_mpc_agent_config = build_pure_mpc_agent_config(cfg)

    # env
    env = gym.make("intersection-v1", render_mode="rgb_array", config=gym_env_config)

    # agent
    mpc_agent = IterativeLinearMPC_Agent(env, pure_mpc_agent_config)

    observation, _ = env.reset()

    for i in range(100):
        # get action from agent
        action = mpc_agent.predict(observation, return_numpy=False)
        # plot the trajectory
        mpc_agent.plot()

        # If your environment expects normalized inputs:
        env_action = [
            action.acceleration / 5.0,
            action.steer / (np.pi / 3.0)
        ]
        observation, reward, done, truncated, info = env.step(env_action)

        # for debug
        print(observation[0][3:5])  # e.g. [speed, heading]

        env.render()

        if done or truncated:
            break

    env.close()


if __name__ == "__main__":
    test_iterative_linear_mpc_agent()
