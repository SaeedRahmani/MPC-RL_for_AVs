import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import gymnasium as gym
from typing import Union
from ..utils.typing import (
    Action, RawObservation, ParsedObservation, 
    Trajectory
)
from ..utils.vehicle_property import VehicleSetting
from ..utils.vehicle_kinematics import Vehicle
from .agent import Agent


class MPC_Agent(Agent):
    """
    The Model Predictive Control Agent.
    """
    def __init__(
        self, 
        env: gym.Env, 
        # dt: float = 0.1, 
        horizon: int = 6,
    ):
        super().__init__(env=env)
        
        """ env default config """
        self.simulation_freq = self.env.config["simulation_frequency"]
        self.policy_frequency = self.env.config["policy_frequency"]
        
        """ MPC time """
        self.dt = 1 / self.policy_frequency # [s] make it aligned with policy frequency
        self.horizon = horizon
        
        """ reference trajectory """
        self.reference_trajectory: Trajectory = self._generate_global_reference_trajectory()
        
        self.ego_vehicle = None
        self.agent_vehicles: Union[list[Vehicle], None] = None
        
    def __str__(self) -> str:
        return f"MPC Agent"
    
    def __repr__(self) -> str:
        return f"MPC Agent"

    def make_decision(self, observation: RawObservation) -> Action:
        """
        Generate decision based on the observation
        
        Argument:
            observation: Observation, a KinematicObservation
            defined in the `highway-env` package.
        
        Return:
            action: Action, a np.ndarray with shape (2,), 
            containing the acceleration and steering angle.
        """
        if observation.shape != self.observation_dim:
            raise ValueError(
                f"Expected action to be of shape {self.observation_dim}, got {observation.shape}")
        
        self._construct_vehicle_objects(observation)        
        mpc_action = self._solve_mpc() 
        action = mpc_action[:2]
        if action.shape != (2,):
            raise ValueError(f"Expected action to be of shape (2,), got {action.shape}")
        return action
    
    def _solve_mpc(self, method: str = "SLSQP", max_iterations: int = 100) -> Action:
        """
        Solve the MPC optimization problem, by find the solution with minimum cost.
        """
        
        # initial guess
        # predict the acceleration and steering angle in the horizon.
        # (1 + 1) * Horizon = 12
        x0=np.zeros(2 * self.horizon)  

        # bounds        
        bounds = [(-10, 3), (-np.pi/2, np.pi/2)] * self.horizon

        # solve by minimizing the cost
        result = minimize(
            fun=self._compute_cost_func,
            x0=x0,
            bounds=bounds,
            method=method, 
            args=(),
            options={'maxiter': max_iterations},
        )   
        
        # only use the action in the next timestamp
        action = result[:2]
        
        return action  
        
    def _construct_vehicle_objects(self, observation: RawObservation) -> None:
        """ Predict the future states of other agents """    
        parsed_observation = []
        
        for agent_index in range(self.max_observed_agents):
            agent_state = observation[agent_index]
            agent_presence = agent_state[0]
            # if agent is presented (equal to 1)
            if agent_presence == 1:
                # parse the row of observation
                agent_state = agent_state[1:]
                parsed_observation.append(agent_state)
        
        parsed_observation = np.stack(parsed_observation)
    
        # ego vehicle
        ego_vehicle_state = parsed_observation[0, :]
        self.ego_vehicle = Vehicle(
            index=0,
            position=ego_vehicle_state[:2],
            speed_x=ego_vehicle_state[2],
            speed_y=ego_vehicle_state[3],
            heading=ego_vehicle_state[4],
        )    
        
        num_agents = parsed_observation.shape[0] - 1 
        if num_agents > 0:
            for agent_index in range(num_agents):
                agent_vehicle_state = parsed_observation[agent_index+1, :]
                self.agent_vehicles.append(
                    Vehicle(
                        index=agent_index+1,
                        position=agent_vehicle_state[:2],
                        speed_x=agent_vehicle_state[2],
                        speed_y=agent_vehicle_state[3],
                        heading=agent_vehicle_state[4],
                    )
                )

    def _compute_cost_func(self, actions, weights) -> float:
        """
        Computes the total cost for the Model Predictive Control (MPC) based on multiple components.

        The cost function includes the following components:
        - `State Cost`: Measures how far the system state deviates from the desired state.
        - `Control Cost`: Measures the magnitude of control inputs to ensure smooth and reasonable control actions.
        - `Obstacle Cost`: Penalizes the distance from obstacles to ensure safe navigation.
        - `Collision Cost`: Penalizes any potential collisions, ensuring the system avoids collisions.
        - `Input Difference Cost`: Penalizes abrupt changes in control inputs to promote smooth transitions.
        - `Final State Cost`:
              
        Arguments:
        - weights (list or array): A list or array of weights corresponding to each cost component.

        Returns:
        - total_cost (float): The total cost
        """
        total_cost = sum([
            self._compute_cost_func_per_timestamp(step, weights, actions) 
            for step in range(self.horizon)
        ])
        return total_cost

    def _compute_cost_func_per_timestep(self, step: int, weights: list, actions) -> float:
        
        # compute six cost components
        cost_components = np.array([
            self._compute_state_cost_per_timestep(step, actions),
            self._compute_control_cost_per_timestep(step, actions),
            self._compute_obstacle_cost_per_timestep(step),
            self._compute_collision_cost_per_timestep(step),
            self._compute_input_diff_cost_per_timestep(self),
            self._compute_final_state_cost_per_timestep(self),
        ])
        
        # compute six weighted cost components
        weights = np.array(weights)
        if weights.shape != cost_components.shape:
            raise ValueError(
                f"The shapes of the costs ({cost_components.shape}) and weights ({weights.shape}) are not same.")
        weighted_costs = np.dot(cost_components, weighted_costs)
        
        # sum up the six components
        total_cost = np.sum(weighted_costs)
        
        return total_cost
                          
    #################
    # Cost components
    #################
    
    def _compute_state_cost_per_timestep(self, step: int) -> float:
        """
        The state_cost in this context is a metric used to evaluate 
        how well the vehicle's state aligns with a reference trajectory. 
        It combines several components:
        
        - `perpendicular deviation` measures how far the vehicle is from 
            the reference path in a direction orthogonal to it; 
        - `parallel deviation` measures how far it is along the direction of the path; 
        - `speed deviation` assesses the difference between the vehicle’s speed and the reference speed; 
        - `heading deviation` evaluates how much the vehicle's heading differs from the reference heading. 
        
        This cost function helps in optimizing the vehicle’s control inputs 
        to minimize these deviations and achieve better trajectory tracking.
        """
        return 0
    
    def _compute_control_cost_per_timestep(self, step: int, action: Action) -> float:
        """
        Compute the control cost for one timestep.
        
        Return:
        - `control_cost` (float): 0.01 * acceleration ** 2 + 0.1 * steer ** 2. 
        """
        control_cost = 0.01 * action[0]**2 + 0.1 * action[1]**2
        return control_cost 
    
    def _compute_obstacle_cost_per_timestep(self, step: int) -> float:
        return 0
    
    def _compute_collision_cost_per_timestep(self, step: int) -> float:
        return 0    
    
    def _compute_input_diff_cost_per_timestep(self, step: int) -> float:
        return 0
    
    def _compute_final_state_cost_per_timestep(self, step: int) -> float:
        return 0
    
    def _generate_global_reference_trajectory(
        self, 
        num_points_before_turning: int = 40,
        num_points_during_turning: int = 20,
        num_points_after_turning: int = 25,
        speed_override_from_RL: float = None
    ) -> Trajectory:
        trajectory = []
        x, y, v, heading, v_ref = 2, 50, 10, -np.pi/2,10  # Starting with 10 m/s speed
        turn_start_y = 20
        radius = 5  # Radius of the curve
        turn_angle = np.pi / 2  # Total angle to turn (90 degrees for a left turn)

        # Go straight until reaching the turn start point
        for _ in range(num_points_before_turning):
            #if collision_points and (x, y) in collision_points:
            v_ref = speed_override_from_RL if speed_override_from_RL is not None else 10  # Set speed based on DRL agent's decision
            x += 0
            y += v * self.dt * np.sin(heading)
            trajectory.append(np.array([x, y, v_ref, heading]))
        
        # Compute the turn
        angle_increment = turn_angle / 20  # Divide the turn into 20 steps
        for _ in range(num_points_during_turning):
            #if collision_points and (x, y) in collision_points:
            v_ref = speed_override_from_RL if speed_override_from_RL is not None else 10  # Set speed based on DRL agent's decision
            heading += angle_increment  # Decrease heading to turn left
            x -= v * self.dt * np.cos(heading)
            y += v * self.dt * np.sin(heading)
        
            trajectory.append(np.array([x, y, v_ref, heading]))
        
        # Continue straight after the turn
        for _ in range(num_points_after_turning):  # Continue for a bit after the turn
            #if collision_points and (x, y) in collision_points:
            v_ref = speed_override_from_RL if speed_override_from_RL is not None else 10  # Set speed based on DRL agent's decision
            x -= v * self.dt * np.cos(heading)
            y += 0
        
            trajectory.append(np.array([x, y, v_ref, heading]))
        
        trajectory = np.stack(trajectory)
        num_points_total = num_points_before_turning + num_points_during_turning + num_points_after_turning
        if trajectory.shape != (num_points_total, 4):
            raise ValueError(f"Expected trajectory to be of shape {trajectory.shape}, got {trajectory.shape}")

        return trajectory
    
    def plot(self, width: float = 100, height: float = 100):
        plt.figure(figsize=(5, 5))
        
        # draw the reference trajectory
        plt.scatter(self.reference_trajectory[:,0], self.reference_trajectory[:,1], 
                    c=VehicleSetting.trajectory_color, s=1)
        
        # draw the ego 
        plt.scatter(x=self.parsed_observation[0, 0],
                    y=self.parsed_observation[0, 1], 
                    c=VehicleSetting.ego_color)
        
        # draw the agents
        num_observed_agents = self.parsed_observation.shape[0] - 1
        if num_observed_agents > 0:
            for agent_index in range(num_observed_agents):
                plt.scatter(x=self.parsed_observation[agent_index+1, 0],
                            y=self.parsed_observation[agent_index+1, 1],
                            c=VehicleSetting.agent_color)   
        
        plt.axis('on')  
        plt.xlim([-width, width])
        plt.ylim([-height, height])

        # flip y-axis
        plt.gca().invert_yaxis()

        plt.show() 