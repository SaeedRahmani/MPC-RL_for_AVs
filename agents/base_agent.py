""" Base class for all the MPC-related agents """

from typing import Union, List, Tuple
import copy

import gymnasium as gym
import numpy as np

from .utils import MPC_Action, Vehicle


class Agent:

    def __init__(
        self, 
        env: gym.Env,
        cfg: dict,
    ) -> None:
        """
        Initializer.
        
        Args:
            env: gym.Env, a highway-env gymnasium environment to retrieve the configuration.
            horizon: int, time horizon parameter in MPC.
            render: bool, whether display the mpc prediction process in a seperate animation.
        """
        # env config
        self.env = env.unwrapped
        self.env_config = self.env.config
        self.config = cfg

        self.simulate_freq: int = self.env_config['simulation_frequency']
        self.policy_freq: int = self.env_config['policy_frequency']
        self.total_vehicles_count = self.env_config["observation"]["vehicles_count"]

        # observation
        self.observed_vehicles_count = 0 # not including ego vehicle
        self.ego_vehicle = None
        self.agent_vehicles = list()
        
        # MPC
        self.horizon = self.config["horizon"]
        self.dt: float = 1 / self.policy_freq # delta T for MPC decision-making               
        self.global_reference_states = self.reference_states
        self.reference_trajectory = self.global_reference_states[:, :2]

        # render
        self.render = self.config["render"]
        self.num_frames_in_dt: int = self.simulate_freq // self.policy_freq

    def __str__(self) -> str:
        return "Base MPC agent"
    
    def predict(
        self, 
        obs: np.ndarray,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, MPC_Action]:

        """
        The high-level method to predict the next action input of vehicle.
        The MPC agent uses this method to interact with the environment.
        
        Args:
            obs: np.ndarray, a collection of KinematicObservation directly received from the interacting environment.
            return_numpy: bool, whether the return value is a np.array or a MPC_Action object.
            
        Returns:
            MPC_Action, including acceleration and steering angle generated by MPC agent.
        """
        self._parse_obs(obs)
        mpc_action = self._solve()
        return mpc_action.numpy() if return_numpy else mpc_action
    
    def _solve(self) -> MPC_Action:
        """
        The low-level core method to solve the MPC problem.
        """
        raise NotImplementedError
    
    def _parse_obs(self, obs: np.ndarray) -> None:
        """
        Parse the observation, a collection of KinematicObservation for MPC modelling.
        
        Args:
            obs: np.ndarray, a collection of KinematicObservation directly received from the interacting environment.
        """
        if not isinstance(obs, np.ndarray):
            raise TypeError(f"Expect observation type np.ndarray, but got {type(obs)}.")
        if obs.shape != (self.total_vehicles_count, 8):
            raise ValueError(f"Expect observation's shape of ({(self.total_vehicles_count, 8)}), but got {obs.shape}")    
    
        self.observed_vehicles_count = np.sum(obs[:, 0] == 1) - 1

        # Ego vehicle
        self.ego_vehicle = Vehicle(
            index=0,
            position=obs[0, 1:3],
            vectorized_speed=obs[0, 3:5],
            heading=self.normalize_angle(obs[0, 5]),
            sinh=obs[0, 6], cosh=obs[0, 7],
        )
        
        # Agent vehicles
        self.agent_vehicles = list()
        if self.observed_vehicles_count > 0:
            for i in range(self.observed_vehicles_count):
                self.agent_vehicles.append(Vehicle(
                    index=i+1,
                    position=obs[i+1, 1:3],
                    vectorized_speed=obs[i+1, 3:5],
                    heading=obs[i+1, 5],
                    sinh=obs[0, 6], cosh=obs[0, 7],
                ))
            assert len(self.agent_vehicles) == self.observed_vehicles_count
        self.agent_vehicles_mpc = copy.deepcopy(self.agent_vehicles)       
    
    @property
    def reference_states(self):
        trajectory = []
        x, y, v, heading, v_ref = 2, 50, 10, -np.pi/2, 10  # Starting with 10 m/s speed
        turn_start_y = 20
        radius = 5  # Radius of the curve
        turn_angle = np.pi / 2  # Total angle to turn (90 degrees for a left turn)

        # Go straight until reaching the turn start point
        for _ in range(40):
            # if collision_points is not None and (x, y) in collision_points:
            #     v = 0  # Set speed based on DRL agent's decision
            x += 0
            y += v * self.dt * np.sin(heading)
            trajectory.append((x, y, v, heading))
        
        # Compute the turn
        angle_increment = turn_angle / 20  # Divide the turn into 20 steps
        for _ in range(20): 
            # if collision_points is not None and (x, y) in collision_points:
            #     v = 0  # Set speed based on DRL agent's decision
            heading -= angle_increment  # Decrease heading to turn left
            x += v * self.dt * np.cos(heading)
            y += v * self.dt * np.sin(heading)
        
            trajectory.append((x, y, v, heading))
        
        # Continue straight after the turn
        for _ in range(25):  # Continue for a bit after the turn
            # if collision_points is not None and (x, y) in collision_points:
            #     v = 0  # Set speed based on DRL agent's decisions
            x += v * self.dt * np.cos(heading)
            y += 0
        
            trajectory.append((x, y, v, heading))
        
        return np.array(trajectory)   
    
    def normalize_angle(self, angle):
        """
        Normalize an angle to the range [-pi, pi].
        
        Parameters:
            angle (float): The angle to be normalized in radians.
            
        Returns:
            float: The normalized angle in the range [-pi, pi].
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def other_vehicle_model(self, other_vehicle, dt):
        new_position = other_vehicle.position + other_vehicle.speed * dt * np.array([np.cos(other_vehicle.heading), np.sin(other_vehicle.heading)])
        return new_position