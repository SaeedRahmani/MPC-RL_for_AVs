import numpy as np

class Action:
    def __init__(
        self, acceleration, steer,
    ) -> None:
        self.acceleration = acceleration
        self.steer = steer
        
    def numpy(self) -> np.ndarray:
        return np.array([self.acceleration, self.steer])
        
class State:
    def __init__(
        self, index, position, speed_xy, heading,
        # angle
    ) -> None:
        # type, shape, range check
        assert index >= 0, f"State.index is not valid"
        assert position.shape == (2,), f"State.position with unexpected shape {position.shape}"
        assert speed_xy.shape == (2,), f"State.speed_xy with unexpected shape {position.shape}"
        # assert angle.shape == (2,), f"State.angle with unexpected shape {position.shape}"
        
        self.index = index
        self.is_ego = True if self.index == 0 else False
        
        # (x, y)
        self.position = position
        # (vx, vy)
        self.speed_xy = speed_xy
        # heading [0, 2pi]
        self.heading = heading
        # (sin_h, cos_h)
        # self.angle = angle  
        
    @property
    def speed(self):
        return np.linalg.norm(self.speed_xy)
        