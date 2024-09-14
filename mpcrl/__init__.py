from gymnasium.envs.registration import register

# register customized gym environments here

# Pure MPC
register(
    id='IntersectionMPC-v0',
    entry_point='mpcrl.env.mpc.IntersectionEnvMPC:IntersectionEnv_MPC',
    max_episode_steps=100,
    reward_threshold=1.0,
)

# MPC + RL