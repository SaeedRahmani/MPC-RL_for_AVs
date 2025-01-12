import hydra
import numpy as np
import gymnasium as gym
import highway_env
from trainers.trainer import DynamicWeightTrainer

from config.config import build_env_config, build_mpcrl_agent_config, build_pure_mpc_agent_config

@hydra.main(config_name="cfg", config_path="./config", version_base="1.3")
def test_mpcrl(cfg):
    # Specify algorithm directly here
    algorithm = "ppo" 
    
    gym_env_config = build_env_config(cfg)
    mpcrl_agent_config = build_mpcrl_agent_config(cfg, version="v1", algorithm=algorithm)
    pure_mpc_agent_config = build_pure_mpc_agent_config(cfg)

    # env
    env = gym.make("intersection-v1", render_mode="human", config=gym_env_config)

    trainer = DynamicWeightTrainer(env, mpcrl_agent_config, pure_mpc_agent_config)
    # trainer.learn()
    # trainer.save()
    trainer.load(
        path=f"./weights/v1/test_{algorithm}_v1",
        mpcrl_cfg=mpcrl_agent_config, 
        version="v1", 
        pure_mpc_cfg=pure_mpc_agent_config,
        env=env,    
    )
    print(trainer.model.policy)
    
    observation, _ = env.reset()
    print(observation)
    for i in range(100):
        action = trainer.predict(observation, False)

        observation, reward, done, truncated, info = env.step([action.acceleration/5, action.steer/(np.pi/3)])
        env.render()
        
if __name__ == "__main__":
    test_mpcrl()