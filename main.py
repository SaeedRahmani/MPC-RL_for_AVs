import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=1, suppress=True)

import gymnasium as gym
import highway_env

from mpcrl.agents.mpc_agent import MPC_Agent

# DOC: https://highway-env.farama.org/observations/index.html#highway_env.envs.common.observation.KinematicObservation
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "heading", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "order": "sorted",
        "normalize": False,
        "absoluate": True,
    },
    "scaling": 2, # scale the rendering animation to show the surrounding vehicles.
}

# initialization
env = gym.make("intersection-v1", render_mode="rgb_array", config=config)

# print(env.observation_space) # 'presence', 'x', 'y', 'vx', 'vy', "heading", "cos_h", "sin_h"
# print(env.unwrapped.config)

mpc_agent = MPC_Agent(env=env)

trajectory = mpc_agent._generate_global_reference_trajectory()
observation, info = env.reset()



# # print(mpc_agent._parse_observation(observation))
# image = env.render()
# mpc_agent.make_decision(observation)
# mpc_agent.plot()








# # main episode game loop
# for _ in range(100):
#     # getting action from agent
#     action = env.action_space.sample()
    
#     # getting action from env
#     state, reward, done, truncated, info = env.step(action)
    
#     # rendering animation
#     env.render()
    
#     # checking end conditions
#     if done or truncated:
#         state = env.reset()

# # destroy all handles
# env.close()