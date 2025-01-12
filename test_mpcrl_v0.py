import os
import glob
import hydra
import numpy as np
import gymnasium as gym
import highway_env
from trainers.trainer import RefSpeedTrainer

from config.config import build_env_config, build_mpcrl_agent_config, build_pure_mpc_agent_config

@hydra.main(config_name="cfg", config_path="./config", version_base="1.3")
def test_mpcrl(cfg):
    # Specify algorithm directly here
    algorithm = "ppo" 
    
    gym_env_config = build_env_config(cfg)
    mpcrl_agent_config = build_mpcrl_agent_config(cfg, version="v0", algorithm=algorithm)
    pure_mpc_agent_config = build_pure_mpc_agent_config(cfg)

    # env
    env = gym.make("intersection-v1", render_mode="human", config=gym_env_config)

    trainer = RefSpeedTrainer(env, mpcrl_agent_config, pure_mpc_agent_config)
    # trainer.learn()
    # trainer.save()
    # Find the latest saved model
    save_dir = "./saved_models"
    model_files = sorted(glob.glob(f"{save_dir}/*"), key=os.path.getmtime, reverse=True)
    if not model_files:
        raise FileNotFoundError(f"No saved models found in {save_dir}")
    
    latest_model_path = model_files[0]  # The most recently saved model
    print(f"Loading latest model: {latest_model_path}")
    
    # Load the latest model
    trainer.load(
        path=latest_model_path,
        mpcrl_cfg=mpcrl_agent_config, 
        version="v0", 
        pure_mpc_cfg=pure_mpc_agent_config,
        env=env,    
    )
    print(trainer.model.policy)
    
    observation, _ = env.reset()
    
    for i in range(100):
        action = trainer.predict(observation, False)
        print('MPC acceleration:', action.acceleration)
        observation, reward, done, truncated, info = env.step([action.acceleration/5, action.steer/(np.pi/3)])
        print('reward:', reward)
        env.render()
        
if __name__ == "__main__":
    test_mpcrl()

#     trainer.load(
#         path=f"./weights/v0/test_{algorithm}_v0",
#         mpcrl_cfg=mpcrl_agent_config, 
#         version="v0", 
#         pure_mpc_cfg=pure_mpc_agent_config,
#         env=env,    
#     )
#     print(trainer.model.policy)
    
#     observation, _ = env.reset()
    
#     for i in range(100):
#         action = trainer.predict(observation, False)
#         print('MPC acceleration:', action.acceleration)
#         observation, reward, done, truncated, info = env.step([action.acceleration/5, action.steer/(np.pi/3)])
#         print('reward:', reward)
#         env.render()
        
# if __name__ == "__main__":
#     test_mpcrl()