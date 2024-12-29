import hydra
import numpy as np
import gymnasium as gym
import highway_env
from trainers.trainer import RefSpeedTrainer

from config.config import build_env_config, build_mpcrl_agent_config, build_pure_mpc_agent_config

@hydra.main(config_name="cfg", config_path="./config", version_base="1.3")
def test_mpcrl(cfg):
    # Specify algorithm directly here
    algorithm = "a2c"  # Change to A2C
    
    # Build configurations
    gym_env_config = build_env_config(cfg)
    mpcrl_agent_config = build_mpcrl_agent_config(cfg, version="v0", algorithm=algorithm)
    pure_mpc_agent_config = build_pure_mpc_agent_config(cfg)

    # Create environment
    env = gym.make("intersection-v1", render_mode="human", config=gym_env_config)

    # Initialize trainer and load saved model
    trainer = RefSpeedTrainer(env, mpcrl_agent_config, pure_mpc_agent_config)
    trainer.load(
        path=f"./weights/v0/test_{algorithm}_v0",  # Ensure this path matches the saved weights
        mpcrl_cfg=mpcrl_agent_config, 
        version="v0", 
        pure_mpc_cfg=pure_mpc_agent_config,
        env=env,    
    )
    print(trainer.model.policy)
    
    # Test the agent in the environment
    observation, _ = env.reset()
    
    for i in range(100):  # Run for 100 steps
        action = trainer.predict(observation, False)
        observation, reward, done, truncated, info = env.step([action.acceleration / 5, action.steer / (np.pi / 3)])
        env.render()
        
if __name__ == "__main__":
    test_mpcrl()
