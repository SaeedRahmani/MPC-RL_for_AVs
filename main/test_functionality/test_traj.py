import hydra
import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from agents.mpcrl_agent import MPCRL_Agent
from agents.pure_mpc import PureMPC_Agent
from config.config import build_env_config, build_pure_mpc_agent_config


@hydra.main(config_name="cfg", config_path="./config", version_base="1.3")
def test_trajectory(cfg):
    gym_env_config = build_env_config(cfg)
    pure_mpc_agent_config = build_pure_mpc_agent_config(cfg)
    env = gym.make("intersection-v1", render_mode="rgb_array", config=gym_env_config)

    mpc_agent = PureMPC_Agent(env, pure_mpc_agent_config)
    
    xyv = mpc_agent.update_reference_states()[:,:3]
    # norm = plt.Normalize(xyv[:, 2].min(), xyv[:, 2].max())
    # colors = cm.viridis(norm(xyv[:, 2]))
    scatter =  plt.scatter(xyv[:, 0], xyv[:, 1], c=xyv[:, 2], cmap='viridis', s=3)
    plt.colorbar(scatter, label='Speed') 
    plt.show()

if __name__ == "__main__":
    test_trajectory()