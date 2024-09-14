import numpy as np
from .typing import Vector2, Action

class Vehicle:
    """
    A basic vehicle model to update its state,
    given action (acceleration and steering).
    """
    
    """ Vehicle length [m] """
    LENGTH = 5.0
    """ Vehicle width [m] """
    WIDTH = 2.0
    
    def __init__(
        self,
        index: int,
        position: Vector2,
        speed_x: float,
        speed_y: float,
        heading: float,
    ):
        self.index = index # 0 for ego vehicle, positive number for agent vehicles 
        self.position = position
        self.speed_x, self.speed_y = speed_x, speed_y
        self.speed = np.linalg.norm(np.array([self.speed_x, self.speed_y]))
        self.heading = heading
        
    def step(self, action: Action) -> None:
        """
        Perform several steps of simulation with constant action.
        """
        frames = int(
            self.config["simulation_frequency"] // self.config["policy_frequency"]
        )
        for frame in range(frames):
            self._simulate_frame(1 / self.config["simulation_frequency"])
        
    def _simulate_frame(self, dt: float, action: Action) -> None:
        """
        Propagate the vehicle state given its actions.
        """
        # TODO: clip action
        delta_f = action["steering"]
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array(
            [np.cos(self.heading + beta), np.sin(self.heading + beta)]
        )
        """ update states """
        self.position += v * dt # new position
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt # new heading
        self.speed += self.action["acceleration"] * dt # new speed
        self.speed_x, self.speed_y = self.speed * np.array([np.cos(self.heading), np.sin(self.heading)]) # new speed conponents