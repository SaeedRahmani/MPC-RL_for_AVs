""" MPC+RL agent """

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env

from .pure_mpc_agent import PureMPC_Agent


class MPCRL_Agent(PureMPC_Agent):
    def __init__(self, env: Env, cfg: dict, version: str = "v0") -> None:
        super().__init__(env, cfg)

        assert version in ["v0", "v1"], "Undefined version of MPCRL Agent"
        
        # self.RL_agent

    def predict(self, obs, return_numpy = True):
        self._parse_obs(obs)
        self.is_collision_detected, collision_points = self.check_potential_collision()
        ref = self.global_reference_states[:, :2]
        
        closest_points = []
        if self.is_collision_detected:
            for point in collision_points:
                distances = np.linalg.norm(ref - point, axis=1) 
                closest_index = np.argmin(distances)
                closest_points.append((ref[closest_index], closest_index))
            self.collision_point_index = min(closest_points, key=lambda x: x[1])[1]
        else:
            self.collision_point_index = None
            
        mpc_action = self._solve()
        return mpc_action.numpy() if return_numpy else mpc_action

    def __str__(self) -> str:
        return """
            MPC+RL agent [Receding Horizon Control], Solved by `CasADi`\n
            RL to generate reference speed or dynamic weights.
        """