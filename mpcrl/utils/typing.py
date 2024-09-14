import numpy as np
from typing import TypeAlias

from highway_env.envs.common.observation import KinematicObservation

Action: TypeAlias = np.ndarray[np.float64]
RawObservation: TypeAlias = KinematicObservation
ParsedObservation: TypeAlias = np.ndarray[np.float64]
Trajectory: TypeAlias = np.ndarray[np.float64]
Vector2: TypeAlias = np.ndarray