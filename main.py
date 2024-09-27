import numpy as np
import gymnasium as gym
import highway_env
from agents.pure_mpc_agent import PureMPC_Agent

np.set_printoptions(suppress=True, precision=2)

# DOC: https://highway-env.farama.org/environments/intersection/#default-configuration
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "heading", "sin_h", "cos_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
            "heading": [-1 * np.pi, np.pi],
            "sin_h": [-1, 1],
            "cos_h": [-1, 1],
        },
        "absolute": True,
        "order": "sorted",
        "normalize": False,
        "absoluate": True,
    },
    "action": {
        "type": "ContinuousAction",
        "steering_range": [-np.pi / 4, np.pi / 4],
        "acceleration_range": [-5.0, 5.0],
        "longitudinal": True,
        "lateral": True,
        "dynamical": True,
    },
    "scaling": 3, # scale the rendering animation to show all the surrounding vehicles.
    "duration": 40,  # [s]
    # "destination": "o1",
    "initial_vehicle_count": -1,
    "spawn_probability": 0.0002,
    "screen_width": 600,
    "screen_height": 600,
    "policy_frequency": 10,
    "simulation_frequency": 30
}

# env
env = gym.make("intersection-v1", render_mode="rgb_array", config=config)

# agent
mpc_agent = PureMPC_Agent(env, horizon=10)

observation, _ = env.reset()
print(mpc_agent.reference_states)

for i in range(100):
    # getting action from agent
    action = mpc_agent.predict(observation, False)
    # mpc_agent.plot()
    # print(np.array([action.acceleration, action.steer]))
    observation, reward, done, truncated, info = env.step([action.acceleration/5, action.steer/(np.pi/3)])
    print(observation[0,:])
    mpc_agent.plot()
    # rendering animation
    env.render()
    
    # checking end conditions
    if done or truncated:
        state = env.reset()

# destroy all handles
env.close()