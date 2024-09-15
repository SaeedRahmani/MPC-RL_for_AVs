import gymnasium as gym
import highway_env
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from highway_env.envs.common.observation import KinematicObservation
from highway_env.vehicle.kinematics import Vehicle
from utils import Action, State

class MPC_Agent():
    """ MPC Agent [Receding Horizon Optimization]"""
    
    def __init__(
        self,
        env: gym.Env,
        horizon: int = 6,
        render: bool = True,
    ):
        self.env = env.unwrapped

        self.simulate_freq: int = self.env.config['simulation_frequency']
        self.policy_freq: int = self.env.config['policy_frequency']
        self.dt: float = 1 / self.policy_freq
        self.num_frames: int = self.simulate_freq // self.policy_freq
    
        self.horizon: int = horizon
        
        # observation
        self.vehicles_count: int = self.env.config['observation']['vehicles_count']
        self.num_features: int = len(self.env.config['observation']['features'])
        
        # render
        if render:
            plt.figure(figsize=(5, 5))
        
    def solve(
        self, 
        observation: KinematicObservation,
        weights: list[float] = None,
        speed_override_from_RL: list[float] = None,
        method: str = "SLSQP",
        max_iterations: int = 100
    ) -> Action:
        """ The high-level method to be called to find the action """
        # parse the observation from env
        self.parseObservation(observation)
        
        # solve the minimization problem
        result = minimize(
            fun=self.cost_function,
            x0=np.zeros(2 * self.horizon),
            bounds=[self.env.config['action']['acceleration_range'], 
                    self.env.config['action']['steering_range']] * self.horizon,
            method=method, 
            args=(weights, speed_override_from_RL),
            options={'maxiter': max_iterations},
        )
        
        # construct the action 
        action = Action(
            acceleration=result.x[0],
            steer=result.x[1])
        return action

    def cost_function(self, actions, weights = None, speed_override_from_RL = None) -> float:
        """ Compute the overall cost in the prediction horizon """
        
        # actions shape check
        assert actions.shape == (self.horizon*2,), f"Unexpected control variable shape {actions.shape}"
        
        # weights key check
        if weights == None:
            # default weights
            weights = {
                "cost_state": 1, 
                "cost_control": 1,
                "cost_obstacle": 1,
                "cost_collision": 1,
                "cost_inputDiff": 1,
                "cost_finalStateDiff": 1000,
            }
        elif isinstance(weights, dict):
            # weights from RL
            assert len(weights) == 6 and set(weights.keys()) == set([
                "cost_state", "cost_control", "cost_obstacle", "cost_collision", "cost_inputDiff", "cost_finalStateDiff"
            ])
        
        # all cost components
        cost_total = 0
        cost_state = 0
        cost_control = 0
        cost_obstacle = 0
        cost_collision = 0
        cost_inputDiff = 0 
        cost_finalStateDiff = 0
        
        ego_state: State = self.vehicle_states[0]
        vehicle_states = copy.deepcopy(self.vehicle_states)
        
        distances = [np.linalg.norm(ego_state.position - np.array(point[:2])) for point in self.reference_trajectory]
        closest_index = np.argmin(distances)

        # loop for the following horizon steps
        for index in range(self.horizon):
            # parse the actions
            action = actions[index*2: (index+1)*2]
            action = Action(acceleration=action[0], steer=action[1])
            
            # update ego state
            future_step_vehicle_states = []
            for vehicle in vehicle_states:
                if vehicle.is_ego:
                    # ego vehicle
                    next_ego_state = self.update_vehicle_state(vehicle, action)
                    future_step_vehicle_states.append(next_ego_state)
                else:
                    # agent vehicle
                    next_state = self.update_vehicle_state(vehicle, Action(0, 0))
                    future_step_vehicle_states.append(next_state)
                       
            # compute state cost
            ref_traj_index = min(closest_index + index, self.reference_trajectory.shape[0] - 1)
            ref_x, ref_y, ref_v, ref_heading = self.reference_trajectory[ref_traj_index,:]
            
            dx = next_ego_state.position[0] - ref_x
            dy = next_ego_state.position[1] - ref_y
            perp_deviation = dx * np.sin(ref_heading) - dy * np.cos(ref_heading)
            para_deviation = dx * np.cos(ref_heading) + dy * np.sin(ref_heading)
            
            cost_state += \
                20 * perp_deviation**2 + \
                1 * para_deviation**2 + \
                1* (next_ego_state.speed - ref_v)**2 + \
                0.1 * (next_ego_state.heading - ref_heading)**2
            
            # compute control cost
            cost_control += 0.01 * action.acceleration**2 + 0.1 * action.steer**2

            # compute obstacle cost TODO
            cost_obstacle += 0
            
            # compute collision cost TODO
            cost_collision += 0

            # compute input diff cost
            if index > 0:
                prev_action = actions[(index-1)*2: index*2]
                prev_action = Action(prev_action[0], prev_action[1])
                cost_inputDiff += 0.5 * (
                    (action.acceleration - prev_action.acceleration)**2 + \
                    (action.steer - prev_action.steer)**2
                )
            else:
                cost_inputDiff += 0
            
            # compute final state cost 
            desired_final_state = self.reference_trajectory[-1,:]
            cost_finalStateDiff += 100 * (
                (next_ego_state.position[0] - desired_final_state[0])**2 + 
                (next_ego_state.position[1] - desired_final_state[1])**2 + 
                (next_ego_state.speed - desired_final_state[2])**2 + 
                (next_ego_state.heading - desired_final_state[3])**2
            )
                        
            ego_state = next_ego_state
            vehicle_states = copy.deepcopy(future_step_vehicle_states)
        
        # sum up the weighted costs in total
        cost_total = \
            weights['cost_state'] * cost_state + \
            weights['cost_control'] * cost_control + \
            weights['cost_collision'] * cost_collision + \
            weights['cost_obstacle'] * cost_obstacle + \
            weights['cost_inputDiff'] * cost_inputDiff + \
            weights['cost_finalStateDiff'] * cost_finalStateDiff
        return cost_total
 
    def parseObservation(self, observation: KinematicObservation):
        """ Take raw observation and save as vehicles' states """
        
        # observation type check
        if isinstance(observation, KinematicObservation):
            raise TypeError(
                f"Expect to receive a KinematicObservation object, but {type(observation)}received"
            )
        # observation shape check
        if observation.shape != (self.vehicles_count, self.num_features):
            raise ValueError(
                f"Expect to receive an array with shape ({self.vehicles_count}, {self.num_features})"
            )
    
        # store the vehicles' states given observation.
        self.vehicle_states = list()
        for vehicle_index in range(self.vehicles_count):
            vehicle_observation = observation[vehicle_index,:]
            vehicle_presence = vehicle_observation[0]
            if vehicle_presence == 1:
                x, y = vehicle_observation[1], vehicle_observation[2]
                vx, vy = vehicle_observation[3], vehicle_observation[4]
                heading = vehicle_observation[5]
                # sinh, cosh = vehicle_observation[6], vehicle_observation[7]
                self.vehicle_states.append(
                    State(
                        index=vehicle_index,
                        position=np.array([x, y]),
                        speed_xy=np.array([vx, vy]),
                        heading=heading,
                        # angle=np.array([sinh, cosh]),
                    )
                )

    def update_vehicle_state(self, vehicle_state: State, action: Action) -> State:
        """ Update the next state of one vehicle based current state and action input. """
        
        # update several frame in one timestep
        for _ in range(self.num_frames):
            delta_f = action.steer
            beta = np.arctan(1 / 2 * np.tan(delta_f))
            v = vehicle_state.speed * np.array(
                [np.cos(vehicle_state.heading + beta), np.sin(vehicle_state.heading + beta)]
            )

            next_vehicle_state = State(
                index=vehicle_state.index,
                position=vehicle_state.position + v * self.dt,
                speed_xy=(vehicle_state.speed + action.acceleration * self.dt) * \
                    np.array([np.cos(vehicle_state.heading), np.sin(vehicle_state.heading)]),
                heading=vehicle_state.heading + vehicle_state.speed * np.sin(beta) / (Vehicle.LENGTH / 2) * self.dt,
            )
            vehicle_state = next_vehicle_state     
        return vehicle_state    
        
    # FIXME:
    @property
    def reference_trajectory(
        self, 
        num_points_before_turning: int = 40,
        num_points_during_turning: int = 20,
        num_points_after_turning: int = 25,
        speed_override_from_RL: float = None
    ):
        trajectory = []
        dt = 0.1
        x, y, v, heading, v_ref = 2, 50, 10, -np.pi/2,10  # Starting with 10 m/s speed
        turn_start_y = 20
        radius = 5  # Radius of the curve
        turn_angle = np.pi / 2  # Total angle to turn (90 degrees for a left turn)

        # Go straight until reaching the turn start point
        for _ in range(num_points_before_turning):
            #if collision_points and (x, y) in collision_points:
            v_ref = speed_override_from_RL if speed_override_from_RL is not None else 10  # Set speed based on DRL agent's decision
            x += 0
            y += v * dt * np.sin(heading)
            trajectory.append(np.array([x, y, v_ref, heading]))

        # Compute the turn
        angle_increment = turn_angle / 20  # Divide the turn into 20 steps
        for _ in range(num_points_during_turning):
            #if collision_points and (x, y) in collision_points:
            v_ref = speed_override_from_RL if speed_override_from_RL is not None else 10  # Set speed based on DRL agent's decision
            heading += angle_increment  # Decrease heading to turn left
            x -= v * dt * np.cos(heading)
            y += v * dt * np.sin(heading)

            trajectory.append(np.array([x, y, v_ref, heading]))

        # Continue straight after the turn
        for _ in range(num_points_after_turning):  # Continue for a bit after the turn
            #if collision_points and (x, y) in collision_points:
            v_ref = speed_override_from_RL if speed_override_from_RL is not None else 10  # Set speed based on DRL agent's decision
            x -= v * dt * np.cos(heading)
            y += 0

            trajectory.append(np.array([x, y, v_ref, heading]))

        trajectory = np.stack(trajectory)
        num_points_total = num_points_before_turning + num_points_during_turning + num_points_after_turning
        if trajectory.shape != (num_points_total, 4):
            raise ValueError(f"Expected trajectory to be of shape {trajectory.shape}, got {trajectory.shape}")

        return trajectory
        
    def plot(self, width: float = 100, height: float = 100):
        plt.clf() 
        # plt.figure(figsize=(5, 5))
        
        # draw the reference trajectory
        plt.scatter(self.reference_trajectory[:,0], self.reference_trajectory[:,1], 
                    c="grey", s=1)
        
        # draw the ego 
        
        for vehicle in self.vehicle_states:
            if vehicle.is_ego:
                plt.scatter(x=vehicle.position[0],
                    y=vehicle.position[1],
                    c="blue")
            else:
                plt.scatter(x=vehicle.position[0],
                    y=vehicle.position[1],
                    c="black")                 
        
        plt.axis('on')  
        plt.xlim([-width, width])
        plt.ylim([-height, height])

        plt.gca().invert_yaxis() # flip y-axis, consistent with pygame window
        
        plt.pause(0.1) # animate
        

if __name__ == "__main__":
    # config
    np.set_printoptions(suppress=True, precision=2)
    
    # DOC: https://highway-env.farama.org/environments/intersection/#default-configuration
    config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "heading", "sin_h", "cos_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
            "heading": [0, 2 * np.pi], # [-1 * np.pi, np.pi],
            "sin_h": [-1, 1],
            "cos_h": [-1, 1],
        },
        "absolute": True,
        "order": "sorted",
        "normalize": False,
        "absoluate": True,
    },
    "action": {
        "type": "ContinuousAction",
        "steering_range": [-np.pi / 3, np.pi / 3],
        "acceleration_range": [-10.0, 3.0],
        "longitudinal": True,
        "lateral": True,
        "dynamical": True,
    },
    "scaling": 3, # scale the rendering animation to show all the surrounding vehicles.
    "duration": 20,  # [s]
    # "destination": "o1",
    "initial_vehicle_count": 10,
    "spawn_probability": 0.2,
    "screen_width": 600,
    "screen_height": 600,
    }
    # env
    env = gym.make("intersection-v1", render_mode="rgb_array", config=config)
    
    # agent
    mpc_agent = MPC_Agent(env, horizon=6)
    # print(env.unwrapped.config)
    
    observation, _ = env.reset()
    print(observation)
    env.render()
     
    action: Action = mpc_agent.solve(observation)
    # print(mpc_agent.reference_trajectory)
    # mpc_agent.plot()
    # plt.show()
    
    for i in range(100):
        # getting action from agent
        action: Action = mpc_agent.solve(observation)
        mpc_agent.plot()
        
        observation, reward, done, truncated, info = env.step(action.numpy())
        
        # rendering animation
        env.render()
        
        # checking end conditions
        if done or truncated:
            state = env.reset()

    # destroy all handles
    env.close()
    