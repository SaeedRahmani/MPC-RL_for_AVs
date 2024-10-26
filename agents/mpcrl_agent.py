""" MPC+RL agent """

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env

from .pure_mpc_agent import PureMPC_Agent


class MPCRL_Agent(PureMPC_Agent):
    ...

    def __str__(self) -> str:
        return """
            MPC+RL agent [Receding Horizon Control], Solved by `CasADi`\n
            RL to generate reference speed or dynamic weights.
        """