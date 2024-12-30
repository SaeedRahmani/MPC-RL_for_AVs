""" Pure MPC agent """

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env
from shapely import LineString

from .base_agent import Agent
from .utils import MPC_Action, Vehicle
import math


class PureMPC_Agent(Agent):
    weight_components = [
        "state", 
        "control", 
        "distance", 
        "collision", 
        "input_diff", 
        "final_state"
    ]

    def __init__(self, env: Env, cfg: dict) -> None:
        """
        Initializer for Pure MPC agent.
        """
        super().__init__(env, cfg)

        # Load collision detection parameters from config
        self.detection_dist = self.config.get("detection_distance", 100)
        self.ttc_threshold = self.config.get("ttc_threshold", 1)
        self.collision_wait_time = 0.3  # Time to wait before resuming original trajectory
        self.collision_timer = 0
        self.resume_original_trajectory = True

        # Load weights for each cost component from config
        self.default_weights = {
            f"weight_{key}": self.config[f"weight_{key}"]
            for key in PureMPC_Agent.weight_components
        }

        # Initialize the matplotlib fig and ax if rendering is enabled
        if self.config.get("render", False):
            window_size = self.config.get("render_window_size", 5)
            self.fig, self.ax = plt.subplots(figsize=(window_size, window_size))
            # Set the position of the figure window (x, y)
            manager = plt.get_current_fig_manager()
            manager.window.move(50, 50)
        
        self.last_acc = 0

        self.original_reference_trajectory = self.generate_global_reference_trajectory(False)
        self.global_reference_trajectory = self.original_reference_trajectory

    def predict(self, obs, return_numpy=True, weights_from_RL=None, ref_speed=None):
        self._parse_obs(obs)
        self._check_collision()

        # Handle collision logic
        if self.is_collide:
            # When collision is detected, modify only the future part of the reference trajectory
            past_trajectory = self.global_reference_trajectory[:self.ego_index]  # Keep the past trajectory
            future_trajectory = self.generate_global_reference_trajectory(self.conflict_points)[self.ego_index:]  # Update only the future
            self.global_reference_trajectory = past_trajectory + future_trajectory
            self.resume_original_trajectory = False
            self.collision_timer = 0  # Reset timer
        elif not self.resume_original_trajectory and not self.is_collide:
            # If no collision is detected, wait for a specific time before resuming original trajectory
            self.collision_timer += self.dt
            if self.collision_timer >= self.collision_wait_time:
                # Resume the original trajectory from current position onwards
                past_trajectory = self.global_reference_trajectory[:self.ego_index]
                future_trajectory = self.original_reference_trajectory[self.ego_index:]
                self.global_reference_trajectory = past_trajectory + future_trajectory
                self.resume_original_trajectory = True

        mpc_action = self._solve(weights_from_RL)
        return mpc_action.numpy() if return_numpy else mpc_action


    def _solve(self, weights_from_RL: dict[str, float] = None) -> MPC_Action:
        """
        Solving the MPC optimization problem with CasADi.
        """
        N = self.horizon  # Time horizon

        # Define symbolic variables for states and controls
        n_states = 4    # [x, y, theta, v]
        n_controls = 2  # [acceleration, steer_angle]

        x = ca.SX.sym('x', n_states, N + 1)  # State trajectory over N + 1 time steps
        u = ca.SX.sym('u', n_controls, N)    # Control inputs over N time steps

        if weights_from_RL is None:
            weights = self.default_weights
        else:
            weights = {
                f"weight_{key}": weights_from_RL[i] 
                for i, key in enumerate(PureMPC_Agent.weight_components)
            }
        
        # Get the index on the reference trajectory for ego vehicle
        self.ego_index = np.argmin(
            [np.linalg.norm(self.ego_vehicle.position - trajectory_point[:2]) 
            for trajectory_point in self.global_reference_trajectory]
        )
        
        # *** Use the updated reference trajectory ***
        # Update the reference trajectory based on collision detection
        ref = np.array(self.global_reference_trajectory)
        closest_index = self.ego_index

        # Define the cost function (objective to minimize)
        total_cost = 0
        state_cost = 0
        control_cost = 0
        distance_cost = 0
        input_diff_cost = 0
        final_state_cost = 0
        
        for k in range(N):
            ref_traj_index = min(closest_index + k, len(ref) - 1)
            ref_x, ref_y, ref_v, ref_heading = ref[ref_traj_index]
            
            # *** Use ref_v from the updated reference trajectory ***
            #ref_v = ref[ref_traj_index, 2]  # Reference speed from updated trajectory
            #ref_heading = ref[ref_traj_index, 3]  # Reference heading
            
            dx = x[0, k] - ref_x
            dy = x[1, k] - ref_y

            perp_deviation = dx * ca.sin(ref_heading) - dy * ca.cos(ref_heading)
            para_deviation = dx * ca.cos(ref_heading) + dy * ca.sin(ref_heading)

            # State cost
            state_cost += (
                10 * perp_deviation**2 + 
                2 * para_deviation**2 +
                5 * (x[3, k] - ref_v)**2 +  # Penalize velocity deviation from ref_v
                0.1 * (x[2, k] - ref_heading)**2  # Penalize heading deviation
            )

            # Control cost
            control_cost += 0.01 * u[0, k]**2 + 0.1 * u[1, k]**2

            # Input difference cost
            if k > 0:
                input_diff_cost += 0.5 * ((u[0, k] - u[0, k-1])**2 + (u[1, k] - u[1, k-1])**2)

            # Distance cost (penalizing closeness to other vehicles)
            # for other_vehicle in self.agent_vehicles_mpc:
            #     dist = ca.norm_2(x[:2, k] - other_vehicle.position)
            #     distance_cost += ca.if_else(
            #         dist < 1.0,
            #         1000 / (dist + 1e-6)**2,
            #         100 / (dist + 1e-6)**2
            #     )

            # Update other vehicles' location (assuming constant speed)
            for other_vehicle in self.agent_vehicles_mpc:
                other_vehicle.position = self.other_vehicle_model(other_vehicle, self.dt)

        # Final state cost
        ref_traj_index = min(closest_index + N, ref.shape[0] - 1)
        desired_final_state = ref[ref_traj_index, :]        
        final_state_cost += 100 * (
            (x[0, -1] - desired_final_state[0])**2 + 
            (x[1, -1] - desired_final_state[1])**2 + 
            (x[3, -1] - desired_final_state[2])**2 +    # Use final ref_v (desired speed)
            (x[2, -1] - desired_final_state[3])**2      # Heading angle
        )

        total_cost = (
            state_cost * weights["weight_state"] +     
            control_cost * weights["weight_control"] +  
            distance_cost * weights["weight_distance"] +
            input_diff_cost * weights["weight_input_diff"] +
            final_state_cost * weights["weight_final_state"]
        )
        if self.is_collide:
            total_cost += 3000 * x[3,k]**2

        # Vehicle dynamics using the Kinematic Bicycle Model
        def vehicle_model(x, u):
            beta = ca.atan(Vehicle.LENGTH_REAR / Vehicle.LENGTH * ca.tan(u[1]))  
            x_next = ca.vertcat(
                x[3] * ca.cos(x[2] + beta),
                x[3] * ca.sin(x[2] + beta),
                (x[3] / Vehicle.LENGTH) * ca.sin(beta),
                u[0]
            )
            return x_next

        # Constraints
        g = []  
        for k in range(N):
            x_next = x[:, k] + vehicle_model(x[:, k], u[:, k]) * self.dt
            g.append(x[:, k + 1] - x_next)  

        g = ca.vertcat(*g)

        opt_variables = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1))

        # Bounds
        lbg = [0] * g.size1()  
        ubg = [0] * g.size1()

        lbx = []
        ubx = []
        for _ in range(N + 1):
            lbx += [-500, -500, -ca.pi, 0]  # Lower bounds [x, y, theta, v]
            ubx += [500, 500, ca.pi, 30]    # Upper bounds [x, y, theta, v]

        for _ in range(N):
            lbx += [-5, -ca.pi / 3]  
            ubx += [5, ca.pi / 3]    

        nlp = {
            'x': opt_variables, 
            'f': total_cost,    
            'g': g,             
        }
        
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 1000,
            'ipopt.tol': 1e-6,
        }
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        state = np.array([self.ego_vehicle.position[0], self.ego_vehicle.position[1], self.ego_vehicle.heading, self.ego_vehicle.speed])
        x0_states = np.tile(state, (N + 1, 1)).flatten()
        u0_controls = np.zeros(n_controls * N)  
        x0 = np.concatenate((x0_states, u0_controls))  

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        
        if not solver.stats()['success']:
            print("NOTICE: No solution found")
            
        opt_values = sol['x'].full()
        u_opt = opt_values[n_states * (N + 1):].reshape((N, n_controls))

        self.last_acc = u_opt[0, 0]

        mpc_action = MPC_Action(acceleration=u_opt[0, 0], steer=u_opt[0, 1])
        return mpc_action

    def _check_collision(self):
        """
        Collision detection logic.
        """
        ego_path_lineString = LineString([(point[0], point[1]) for point in self.global_reference_trajectory])

        ego_location = np.array(self.ego_vehicle.position)
        ego_index = np.argmin([
            np.linalg.norm(ego_location - np.array(trajectory_point[:2])) 
            for trajectory_point in self.global_reference_trajectory
        ])
        self.ego_index = ego_index

        self.agent_current_locations = []
        self.agent_future_locations = []
        self.conflict_points = []
        self.conflict_index = []
        self.agent_collide = []

        for _, agent_veh in enumerate(self.agent_vehicles):
            agent_current_location = np.array(agent_veh.position)
            self.agent_current_locations.append(agent_current_location)

            agent_future_location = agent_current_location + self.detection_dist * np.array([
                np.cos(agent_veh.heading), np.sin(agent_veh.heading)])
            self.agent_future_locations.append(agent_future_location)

            agent_path_lineString = LineString([agent_current_location, agent_future_location])

            intersection = ego_path_lineString.intersection(agent_path_lineString)
            if not intersection.is_empty:
                if intersection.geom_type == 'Point':
                    conflict_location = np.array((intersection.x, intersection.y))
                    agent_index = np.argmin([np.linalg.norm(conflict_location - np.array(point[:2]))
                        for point in self.reference_trajectory])
                    if ego_index < agent_index:
                        agent_dist = np.linalg.norm(agent_current_location - conflict_location)
                        agent_speed = agent_veh.speed
                        agent_eta = agent_dist / agent_speed

                        ego_dist = LineString([(point[0], point[1]) for point in self.global_reference_trajectory[ego_index: agent_index+1]]).length

                        ego_eta = self.calculate_ego_eta(ego_dist, ego_index, self.last_acc, max_speed=10) 

                        if np.abs(ego_eta - agent_eta) < self.ttc_threshold:
                            print(np.abs(ego_eta - agent_eta), agent_eta)
                            self.agent_collide.append(True)
                            self.conflict_points.append((intersection.x, intersection.y))
                            self.conflict_index.append(agent_index)
                        else:
                            self.agent_collide.append(False)
                            self.conflict_points.append(None)
                    else:
                        self.conflict_points.append(None)
                        self.agent_collide.append(False)
                else:
                    self.conflict_points.append(None)
                    self.agent_collide.append(False)
            else:
                self.conflict_points.append(None)
                self.agent_collide.append(False)

        self.is_collide = np.any(self.agent_collide)
        

    def calculate_ego_eta(self, distance, ego_index, acceleration: float = 1, max_speed=10):
        """
        Calculates estimated time-to-arrival for the ego vehicle.
        """
        current_speed = self.ego_vehicle.speed
        
        velocity_vector = np.array([
            current_speed * np.cos(self.ego_vehicle.heading),
            current_speed * np.sin(self.ego_vehicle.heading),
        ])
        
        trajectory_vector = np.array([
            max_speed * np.cos(self.reference_states[ego_index, 3]),
            max_speed * np.sin(self.reference_states[ego_index, 3]),
        ])

        sign = 1 if np.dot(velocity_vector, trajectory_vector) > 0 else -1
        adjusted_speed = sign * current_speed

        if adjusted_speed >= max_speed:
            return distance / max_speed
        
        if acceleration > 0:
            distance_to_reach_vmax = (max_speed**2 - adjusted_speed**2) / (2 * acceleration)
        else:
            return np.inf  

        if distance > distance_to_reach_vmax:
            time_to_accelerate = (max_speed - adjusted_speed) / acceleration
            return time_to_accelerate + (distance - distance_to_reach_vmax) / max_speed
        else:
            achievable_speed = np.sqrt(2 * acceleration * distance + adjusted_speed**2)
            return achievable_speed

    
    
    def generate_global_reference_trajectory(self,collision_points=None, speed_override=None): 
        trajectory = []
        dt = 0.1
        x, y, v, heading, v_ref = 2, 50, 10, -np.pi/2,10  # Starting with 10 m/s speed
        turn_start_y = 20
        radius = 5  # Radius of the curve
        turn_angle = np.pi / 2  # Total angle to turn (90 degrees for a left turn)

        # Go straight until reaching the turn start point
        for _ in range(40):
            if collision_points:
                v =  0  # Set speed based on DRL agent's decision
                print(v)
            x += 0
            y += v * dt * math.sin(heading)
            trajectory.append((x, y, v, heading))
        
        # Compute the turn
        angle_increment = turn_angle / 20  # Divide the turn into 20 steps
        for _ in range(20):
            if collision_points :
                v =  0  # Set speed based on DRL agent's decision
                print(v)
            heading += angle_increment  # Decrease heading to turn left
            x -= v * dt * math.cos(heading)
            y += v * dt * math.sin(heading)
        
            trajectory.append((x, y, v, heading))
        
        # Continue straight after the turn
        for _ in range(25):  # Continue for a bit after the turn
            if collision_points:
                v =  0  # Set speed based on DRL agent's decision
                print(v)
            
            x -= v * dt * math.cos(heading)
            y += 0
        
            trajectory.append((x, y, v, heading))
        
        return trajectory

    
    def plot(self):
        """ Visualize the MPC solving process """
        x_range = self.config.get("render_axis_range", 50)
        
        # Clear the content of the last frame
        self.ax.clear()
        
        # Set limits and invert y-axis
        self.ax.set_xlim([-x_range, x_range])
        self.ax.set_ylim([-x_range, x_range])  # y_range is set the same as x_range
        self.ax.invert_yaxis()
        self.ax.grid(True)

        # Set labels and title
        self.ax.set_xlabel("X (meter)")
        self.ax.set_ylabel("Y (meter)")
        self.ax.set_title("Pure MPC Agent")

        # Plot the past reference trajectory in grey (unchanged part)
        self.ax.scatter(
            x=[point[0] for point in self.global_reference_trajectory[:self.ego_index]], 
            y=[point[1] for point in self.global_reference_trajectory[:self.ego_index]], 
            color='grey', 
            s=1,
            label="Past Reference"
        )
        
        # Plot the future reference trajectory in green (changed after collision)
        self.ax.scatter(
            x=[point[0] for point in self.global_reference_trajectory[self.ego_index:]], 
            y=[point[1] for point in self.global_reference_trajectory[self.ego_index:]], 
            color='green', 
            s=1,
            label="Future Reference"
        )
        
        # Plot the ego vehicle's location in blue
        ego_x, ego_y = self.ego_vehicle.position
        self.ax.scatter(
            x=ego_x, 
            y=ego_y, 
            color='blue', 
            s=5,
            label="Ego Vehicle"
        )

        # Plot the agent vehicles' locations in red
        for agent_vehicle in self.agent_vehicles:
            agent_x, agent_y = agent_vehicle.position
            self.ax.scatter(
                x=agent_x, 
                y=agent_y, 
                color='red', 
                s=5,
                label="Agent Vehicles"
            )

        # Plot the potential collisions
        for current_loc, conflict_loc in zip(self.agent_current_locations, self.conflict_points):
            if conflict_loc is not None:
                xs = [current_loc[0], conflict_loc[0]]
                ys = [current_loc[1], conflict_loc[1]]
                self.ax.plot(xs, ys, 'x-', markersize=5)

        plt.legend()
        plt.pause(0.1)


    