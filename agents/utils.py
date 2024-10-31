import numpy as np
import math

class MPC_Action:
    def __init__(
        self, acceleration, steer
    ) -> None:
        self.acceleration = acceleration
        self.steer = steer
        
    def numpy(self) -> np.ndarray:
        return np.array([self.acceleration, self.steer])
    
    
     
class Vehicle:
    
    LENGTH = 2.5 # wheelbase
    LENGTH_REAR = LENGTH / 2 # Distance from rear axle to center of mass
        
    def __init__(
        self,
        index,
        position,
        vectorized_speed,
        heading,
        sinh: float,
        cosh: float,
    ):
        self.index = index
        self.is_ego = True if index == 0 else False
        
        self.position = position
        self.vectorized_speed = vectorized_speed
        self.heading = heading
        self.speed = np.linalg.norm(self.vectorized_speed)
        self.sinh = sinh
        self.cosh = cosh
        self.max_acceleration = 3.5
        self.max_deceleration = -10
        
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
    
def generate_global_reference_trajectory(collision_points=None, speed_override=None, dt=0.1) -> list:
    """
    Generate the reference trajectory of the ego-vehicle in the global coordinate system.
    This reference trajectory can be modified based on the predicted collision point. 
    If there is a collision, the reference speeds after that point be zeroed out.

    Args:
        collision_points: Predicted potential collision point. Defaults to None.

    Returns:
        List: A list of points representing the reference trajectory, with x, y, heading angle and speed. 
    """
    
    trajectory = []
    x, y, v, heading, v_ref = 2, 50, 10, -np.pi/2, 10  # Starting with 10 m/s speed
    turn_start_y = 20
    radius = 5  # Radius of the curve
    turn_angle = np.pi / 2  # Total angle to turn (90 degrees for a left turn)

    # Go straight until reaching the turn start point
    for _ in range(40):
        if collision_points and (x, y) in collision_points:
            v = 0  # Set speed based on DRL agent's decision
        x += 0
        y += v * dt * math.sin(heading)
        trajectory.append((x, y, v, heading))
    
    # Compute the turn
    angle_increment = turn_angle / 20  # Divide the turn into 20 steps
    for _ in range(20):
        if collision_points and (x, y) in collision_points:
            v = 0  # Set speed based on DRL agent's decision
        heading += angle_increment  # Decrease heading to turn left
        x -= v * dt * math.cos(heading)
        y += v * dt * math.sin(heading)
     
        trajectory.append((x, y, v, heading))
    
    # Continue straight after the turn
    for _ in range(25):  # Continue for a bit after the turn
        if collision_points and (x, y) in collision_points:
            v = 0  # Set speed based on DRL agent's decision
        
        x -= v * dt * math.cos(heading)
        y += 0
       
        trajectory.append((x, y, v, heading))
    
    return trajectory