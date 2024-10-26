import hydra
import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from agents.mpcrl_agent import MPCRL_Agent
from agents.pure_mpc_agent import PureMPC_Agent
from config.config import build_env_config, build_pure_mpc_agent_config


@hydra.main(config_name="cfg", config_path="./config", version_base="1.3")
def test_trajectory(cfg):
    gym_env_config = build_env_config(cfg)
    pure_mpc_agent_config = build_pure_mpc_agent_config(cfg)
    env = gym.make("intersection-v1", render_mode="rgb_array", config=gym_env_config)

    mpc_agent = PureMPC_Agent(env, pure_mpc_agent_config)
    
    xyv = mpc_agent.reference_states[:,:3]
    norm = plt.Normalize(xyv[:, 2].min(), xyv[:, 2].max())
    colors = cm.viridis(norm(xyv[:, 2]))
    scatter = plt.scatter(xyv[:, 0], xyv[:, 1], c=colors, s=3)  # 's' 控制点的大小
    plt.colorbar(scatter, label='Speed') 
    plt.show()
    print(xyv)
if __name__ == "__main__":
    test_trajectory()