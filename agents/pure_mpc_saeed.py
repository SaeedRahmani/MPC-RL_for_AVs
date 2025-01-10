""" Pure MPC agent """

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env
from shapely import LineString

from .base_agent import Agent
from .utils import MPC_Action, Vehicle


class PureMPC_Agent(Agent):

    weight_components = [
        "state", 
        "control", 
        "distance", 
        "collision", 
        "input_diff", 
        "final_state"
    ]

    def __init__(
        self, 
        env: Env,
        cfg: dict, 
    ) -> None:
        """
        Initializer.
        
        Args:
            env: gym.Env, a highway-env gymnasium environment to retrieve the configuration.
            cfg: configuration dict for pure mpc agent.
        """
        super().__init__(env, cfg)

        self.collision_memory = 0  # Add collision memory counter
        self.collision_memory_steps = 10  # How many steps to remember collision

        # Load collision detection parameters from config
        # self.detection_dist = self.config.get("detection_distance", 100)
        self.ttc_threshold = self.config.get("ttc_threshold", 3)

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
            manager.window.wm_geometry("+50+50")  
        
        self.last_acc = 0

    def __str__(self) -> str:
        return "Pure MPC agent [Receding Horizon Control], Solved by `CasADi` "
      
    def predict(
            self, 
            obs, 
            return_numpy = True,
            weights_from_RL = None,
            ref_speed = None,
        ):
        self._parse_obs(obs)
        self._check_collision()
        mpc_action = self._solve(weights_from_RL, ref_speed)
        return mpc_action.numpy() if return_numpy else mpc_action
      
    def _solve(self, weights_from_RL: dict[str, float]=None, ref_speed_from_RL=None) -> MPC_Action:
        
        manual_collision_avoidance = True
        # MPC parameters
        N = self.horizon
        
        # Define symbolic variables for states and controls
        n_states = 4    # [x, y, theta, v]
        n_controls = 2  # [acceleration, steer_angle]
        
        # Create symbolic variables for state and control trajectories
        x = ca.SX.sym('x', n_states, N + 1)  # State trajectory over N + 1 time steps
        u = ca.SX.sym('u', n_controls, N)    # Control inputs over N time steps

        """ Update weights """
        if weights_from_RL is None:
            # Use default weights from configuration file
            weights = self.default_weights
        else:
            # Use dynamic weights from RL agent
            weights = {
                f"weight_{key}": weights_from_RL[0, i] 
                for i, key in enumerate(PureMPC_Agent.weight_components)
            }

        # Get the index on the reference trajectory for ego vehicle
        self.ego_index = np.argmin(
            [np.linalg.norm(self.ego_vehicle.position - trajectory_point) 
             for trajectory_point in self.reference_trajectory]
        )
        
        # Generate new reference states, given the result of collision detection
        ref = self.update_reference_states(
            speed_override=self.config["speed_override"],
            speed_overide_from_RL=ref_speed_from_RL)
        
        # distances = [np.linalg.norm(self.ego_vehicle.position - np.array(point[:2])) for point in ref]
        closest_index = self.ego_index

        # Define the cost function (objective to minimize)
        total_cost = 0
        state_cost = 0
        control_cost = 0
        distance_cost = 0
        collision_cost = 0
        input_diff_cost = 0
        final_state_cost = 0
        
        for k in range(N):
            ref_traj_index = min(closest_index + k, ref.shape[0] - 1)

            ref_v = ref[ref_traj_index,2]       

            """ Gozde version"""
            dx = x[0, k] - ref[ref_traj_index,0]
            dy = x[1, k] - ref[ref_traj_index,1]

            ref_v = ref[ref_traj_index,2]
            ref_heading = ref[ref_traj_index,3]
            perp_deviation = dx * ca.sin(ref_heading) - dy * ca.cos(ref_heading)
            para_deviation = dx * ca.cos(ref_heading) + dy * ca.sin(ref_heading)
            
            speed_weight = 1
            if self.is_collide:
                # print('collision', self.is_collide)
                # print('ref v', ref_v)
                speed_weight = 100
                
            # State cost
            state_cost += (
                4 * perp_deviation**2 + 
                2 * para_deviation**2 +
                speed_weight * (x[3, k] - ref_v)**2 + 
                # speed_weight * x[3, k]**2 +
                0.5 * (x[2, k] - ref_heading)**2
            )

            # Control cost
            control_cost += 0.01 * u[0, k]**2 + 0.01 * u[1, k]**2
            
            # Input difference cost
            if k > 0:
                input_diff_cost += 0.01 * ((u[0, k] - u[0, k-1])**2 + (u[1, k] - u[1, k-1])**2)

            ### This cost must be zero if we want to mannually change the ref traj in case of accidents
            # Distance cost
            
            if not manual_collision_avoidance:
                for other_vehicle in self.agent_vehicles_mpc:
                    dist = ca.norm_2(x[:2, k] - other_vehicle.position)
                    # in casadi, use ca.if_else to branch
                    distance_cost += ca.if_else(
                        dist < 1.0, # if-statement
                        1000 / (dist + 1e-6)**2,  # if True 
                        100 / (dist + 1e-6)**2    # if False
                    )
            
                collision_cost += ca.if_else(
                    self.is_collide,
                    3000 * x[3, k] ** 2,
                    0
                )
            else:    
                distance_cost = 0
                collision_cost = 0

            # Update other vehicles' location (constant speed)
            for other_vehicle in self.agent_vehicles_mpc:
                other_vehicle.position = self.other_vehicle_model(other_vehicle, self.dt)
        

        # final state cost
        ref_traj_index = min(closest_index + N, ref.shape[0] - 1)
        desired_final_state = ref[ref_traj_index, :]        
        final_state_cost += 100 * (
            (x[0, -1] - desired_final_state[0])**2 + 
            (x[1, -1] + desired_final_state[1])**2 + 
            20 * (x[3, -1] - desired_final_state[2])**2 +    # ref speed
            (x[2, -1] - desired_final_state[3])**2      # heading angle
        )

        total_cost = (
            state_cost * weights["weight_state"] +      # old weight: 10
            control_cost * weights["weight_control"] +  # old weight: 1
            distance_cost * weights["weight_distance"] +
            collision_cost * weights["weight_collision"] + 
            input_diff_cost * weights["weight_input_diff"] +
            final_state_cost * weights["weight_final_state"]
        )

        # Define a function to evaluate each cost component based on the optimized state and control inputs
        cost_fn = ca.Function('cost_fn', [ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1))],
                              [state_cost, control_cost, final_state_cost, input_diff_cost, distance_cost, collision_cost])


        # Define the vehicle dynamics using the Kinematic Bicycle Model
        def vehicle_model(x, u):
            beta = ca.atan(Vehicle.LENGTH_REAR / Vehicle.LENGTH * ca.tan(u[1]))  # Slip angle
            x_next = ca.vertcat(
                x[3] * ca.cos(x[2] + beta),                 # x_dot
                x[3] * ca.sin(x[2] + beta),                 # y_dot
                (x[3] / Vehicle.LENGTH) * ca.sin(beta),     # theta_dot
                u[0]                                        # v_dot (acceleration)
            )
            return x_next
        
        # Constraints
        g = []  # Constraints vector
        
        state = np.array([
            self.ego_vehicle.position[0], 
            self.ego_vehicle.position[1], 
            self.ego_vehicle.heading, 
            self.ego_vehicle.speed
        ])

        x0_states = np.tile(state, (N + 1, 1)).flatten()  # Shape: (4 * (N + 1),)
        # Initial guess for the optimization
        # state = np.array([self.ego_vehicle.position[0], self.ego_vehicle.position[1], self.ego_vehicle.heading, self.ego_vehicle.speed])
        # x0_states = np.tile(state, (N + 1, 1)).flatten()  # Shape: (4 * (N + 1),)
        u0_controls = np.zeros(n_controls * N)           # Shape: (2 * N,)
        # Combine the initial guesses for states and controls
        x0 = np.concatenate((x0_states, u0_controls))    # Shape: (6 * N + 4,)

        # Initial condition constraint
        g.append(x[:, 0] - state)
    
        # State-update constraints for the entire horizon
        for k in range(N):
            x_next = x[:, k] + vehicle_model(x[:, k], u[:, k]) * self.dt
            g.append(x[:, k + 1] - x_next)  # Ensure next state matches dynamics
        
        # Flatten the list of constraints
        g = ca.vertcat(*g)

        # Optimization variables (state and control inputs combined)
        opt_variables = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1))

        # Define bounds on the states and control inputs
        lbg = [0] * g.size1()  # Equality constraints for dynamics
        ubg = [0] * g.size1()

        # Define bounds on control inputs
        lbx = []
        ubx = []
        
        # Bounds on the state variables (no specific bounds for now)
        for _ in range(N + 1):
            lbx += [-500, -500, -ca.pi, 0]  # Lower bounds [x, y, theta, v]
            ubx += [500, 500, ca.pi, 30]    # Upper bounds [x, y, theta, v]
            # lbx += [-ca.inf, -ca.inf, -ca.pi, 0]  # Lower bounds [x, y, theta, v]
            # ubx += [ca.inf, ca.inf, ca.pi, 30]    # Upper bounds [x, y, theta, v]
        # Bounds on control inputs (acceleration and steering angle)
        for _ in range(N):
            lbx += [-5, -ca.pi / 3]  # Lower bounds [acceleration, steer_angle]
            ubx += [5, ca.pi / 3]    # Upper bounds [acceleration, steer_angle]
        
        # Create the optimization problem
        nlp = {
            'x': opt_variables, # decision variables
            'f': total_cost,    # objective (total cost to minimize)
            'g': g,             # constraint
        }
        
        opts = {
            'ipopt.print_level': 0, 
            'print_time':0,
            'ipopt.max_iter':1000,
            'ipopt.tol':1e-6,
        }
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # Solve the optimization problem
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        
        if not solver.stats()['success']:
            print("NOTICE: Not found solution")
            
        
        # # Solve the optimization problem
        opt_values = sol['x'].full()
        # print('optimal values', opt_values)
        u_opt = sol['x'][-N * n_controls:].full().reshape(N, n_controls)
        # print('optimal input values', u_opt)
        # print('first acc', u_opt[0, 0])

        self.last_acc = u_opt[0, 0]

        mpc_action = MPC_Action(acceleration=u_opt[0, 0], steer=u_opt[0, 1])
        return mpc_action
    

    def visualize_predictions(self):
        """Visualize predictions for debugging purposes."""
        x_range = self.config.get("render_axis_range", 50)
        
        # Clear the content of the last frame
        self.ax.clear()
        
        # Set limits and invert y-axis
        self.ax.set_xlim([-x_range, x_range])
        self.ax.set_ylim([-x_range, x_range])
        self.ax.invert_yaxis()
        self.ax.grid(True)

        # Set labels and title
        self.ax.set_xlabel("X (meter)")
        self.ax.set_ylabel("Y (meter)")
        self.ax.set_title("Trajectory Predictions Debug")

        # Plot reference trajectory
        self.ax.plot(
            self.reference_trajectory[:, 0],
            self.reference_trajectory[:, 1],
            'k--',
            alpha=0.5,
            label='Reference trajectory'
        )
        
        # Get ego predictions
        ego_pred = self.predict_ego_future_positions(
            self.ego_vehicle.position,
            self.ego_vehicle.speed,
            self.ego_vehicle.heading,
            self.ego_vehicle.max_acceleration,
            self.dt,
            30,  # prediction_horizon
            self.reference_states[self.ego_index, 2]  # reference_speed
        )
        ego_pred = np.array(ego_pred)
        
        # Plot ego vehicle current position and predictions
        self.ax.scatter(
            self.ego_vehicle.position[0],
            self.ego_vehicle.position[1],
            color='blue',
            s=100,
            marker='o',
            label='Ego vehicle'
        )
        self.ax.plot(
            ego_pred[:, 0],
            ego_pred[:, 1],
            'b-',
            linewidth=2,
            alpha=0.7,
            label='Ego prediction'
        )
        
        # Plot arrow for ego vehicle heading
        heading_length = 2.0
        self.ax.arrow(
            self.ego_vehicle.position[0],
            self.ego_vehicle.position[1],
            heading_length * np.cos(self.ego_vehicle.heading),
            heading_length * np.sin(self.ego_vehicle.heading),
            head_width=0.5,
            head_length=0.8,
            fc='blue',
            ec='blue',
            alpha=0.5
        )
        
        # Plot other vehicles and their predictions
        for i, vehicle in enumerate(self.agent_vehicles):
            # Get predictions for this vehicle
            pred = self.predict_future_positions(
                vehicle.position,
                vehicle.speed,
                vehicle.heading,
                self.dt,
                30  # prediction_horizon
            )
            pred = np.array(pred)
            
            # Plot current position
            self.ax.scatter(
                vehicle.position[0],
                vehicle.position[1],
                color='red',
                s=100,
                marker='o',
                label='Other vehicle' if i == 0 else ""
            )
            
            # Plot predicted trajectory
            self.ax.plot(
                pred[:, 0],
                pred[:, 1],
                'r-',
                alpha=0.5,
                linewidth=2,
                label='Other prediction' if i == 0 else ""
            )
            
            # Plot arrow for vehicle heading
            self.ax.arrow(
                vehicle.position[0],
                vehicle.position[1],
                heading_length * np.cos(vehicle.heading),
                heading_length * np.sin(vehicle.heading),
                head_width=0.5,
                head_length=0.8,
                fc='red',
                ec='red',
                alpha=0.5
            )
        
        # Add legend
        self.ax.legend()
        
        # Print debug information
        debug_info = f"""
        Ego Vehicle:
            Position: ({self.ego_vehicle.position[0]:.2f}, {self.ego_vehicle.position[1]:.2f})
            Speed: {self.ego_vehicle.speed:.2f}
            Heading: {self.ego_vehicle.heading:.2f}
            Reference Speed: {self.reference_states[self.ego_index, 2]:.2f}
        """
        self.ax.text(
            -x_range + 2,
            -x_range + 5,
            debug_info,
            bbox=dict(facecolor='white', alpha=0.7),
            family='monospace'
        )
        
        # Pause briefly to update plot
        plt.pause(0.1)
    
    def predict_ego_future_positions(self, current_position, speed, heading, max_acceleration, dt, prediction_horizon, reference_speed):
        """
        Predict ego vehicle future positions based strictly on reference trajectory points.
        Handles speed adjustments while following the reference path.
        
        Returns:
            list: List of predicted positions [(x1,y1), (x2,y2), ...]
        """
        future_positions = [current_position]
        current_speed = speed
        
        # Find starting index on reference trajectory
        start_index = np.argmin([
            np.linalg.norm(current_position - np.array(point[:2])) 
            for point in self.reference_trajectory
        ])
        
        # Calculate cumulative distances along reference trajectory
        ref_points = self.reference_trajectory[start_index:, :2]
        if len(ref_points) < 2:
            return future_positions
            
        cumulative_distances = [0]
        for i in range(1, len(ref_points)):
            d = np.linalg.norm(ref_points[i] - ref_points[i-1])
            cumulative_distances.append(cumulative_distances[-1] + d)
        
        # For each prediction step
        current_distance = 0
        
        for _ in range(prediction_horizon):
            # Update speed based on reference speed
            if current_speed < reference_speed:
                current_speed = min(current_speed + max_acceleration * dt, reference_speed)
            else:
                current_speed = reference_speed
                
            # Calculate distance traveled in this time step
            current_distance += current_speed * dt
            
            # Find the reference points we're between
            next_idx = np.searchsorted(cumulative_distances, current_distance)
            if next_idx >= len(ref_points):
                # If we've gone beyond the reference trajectory, stop here
                break
                
            if next_idx == 0:
                # We're still near the start
                next_position = ref_points[0]
            else:
                # Interpolate between reference points
                prev_idx = next_idx - 1
                prev_point = ref_points[prev_idx]
                next_point = ref_points[next_idx]
                
                # Calculate interpolation factor
                prev_dist = cumulative_distances[prev_idx]
                next_dist = cumulative_distances[next_idx]
                alpha = (current_distance - prev_dist) / (next_dist - prev_dist) if next_dist != prev_dist else 1.0
                alpha = np.clip(alpha, 0, 1)
                
                # Interpolate position
                next_position = prev_point + alpha * (next_point - prev_point)
                
            future_positions.append(next_position)
        
        return future_positions

    def predict_future_positions(self, current_position, speed, heading, dt, prediction_horizon):
        """
        Predict the future positions of a vehicle based on its current speed and heading.

        Args:
            current_position (np.ndarray): Current position [x, y] of the vehicle.
            speed (float): Current speed of the vehicle.
            heading (float): Heading angle of the vehicle in radians.
            dt (float): Time step for prediction.
            prediction_horizon (int): Number of steps to predict into the future.

        Returns:
            list: A list of future positions [x, y] at each time step.
        """
        future_positions = [current_position]
        for _ in range(prediction_horizon):
            next_position = future_positions[-1] + speed * dt * np.array([
                np.cos(heading),
                np.sin(heading)
            ])
            future_positions.append(next_position)
        return future_positions

    def _check_collision(self):
        """
        Collision detection based on trajectory intersection and time threshold.
        """
        PREDICTION_HORIZON = 30
        TIME_THRESHOLD = 30  # time steps (equivalent to TIME_THRESHOLD * dt seconds)
        
        # Get the current position of the ego vehicle
        ego_location = np.array(self.ego_vehicle.position)

        # Find closest point on reference trajectory
        self.ego_index = np.argmin([
            np.linalg.norm(ego_location - np.array(trajectory_point)) 
            for trajectory_point in self.reference_trajectory
        ])

        # Predict ego future positions
        ego_future_positions = self.predict_ego_future_positions(
            current_position=self.ego_vehicle.position,
            speed=self.ego_vehicle.speed,
            heading=self.ego_vehicle.heading,
            max_acceleration=self.ego_vehicle.max_acceleration,
            dt=self.dt,
            prediction_horizon=PREDICTION_HORIZON,
            reference_speed=self.reference_states[self.ego_index, 2]
        )

        # Create LineString for ego trajectory
        ego_path = LineString(ego_future_positions)

        # Initialize storage for visualization and conflict tracking
        self.agent_current_locations = []
        self.agent_future_locations = []
        self.conflict_points = []
        self.conflict_index = []
        self.agent_collide = []

        for agent_veh in self.agent_vehicles:
            # Store current location
            agent_current_location = np.array(agent_veh.position)
            self.agent_current_locations.append(agent_current_location)

            # Predict future positions
            agent_future_positions = self.predict_future_positions(
                current_position=agent_current_location,
                speed=agent_veh.speed,
                heading=agent_veh.heading,
                dt=self.dt,
                prediction_horizon=PREDICTION_HORIZON
            )
            self.agent_future_locations.append(agent_future_positions)

            # Create LineString for agent path
            agent_path = LineString(agent_future_positions)

            # Check for intersection
            intersection = ego_path.intersection(agent_path)
            
            collision_detected = False
            intersection_point = None
            conflict_idx = None

            if not intersection.is_empty:
                # Extract intersection points based on geometry type
                intersection_points = []
                
                if intersection.geom_type == 'Point':
                    intersection_points.append((intersection.x, intersection.y))
                elif intersection.geom_type == 'LineString':
                    # Take midpoint of the line
                    coords = list(intersection.coords)
                    if coords:
                        mid_idx = len(coords) // 2
                        intersection_points.append(coords[mid_idx])
                elif intersection.geom_type == 'MultiPoint':
                    # Handle multiple intersection points
                    for point in intersection.geoms:
                        intersection_points.append((point.x, point.y))
                elif intersection.geom_type == 'MultiLineString':
                    # Handle multiple line intersections
                    for line in intersection.geoms:
                        coords = list(line.coords)
                        if coords:
                            mid_idx = len(coords) // 2
                            intersection_points.append(coords[mid_idx])
                
                # Check each intersection point
                for int_point in intersection_points:
                    intersection_point = np.array(int_point)
                    
                    # Find closest points in both trajectories to intersection
                    ego_times = [i for i in range(len(ego_future_positions))]
                    agent_times = [i for i in range(len(agent_future_positions))]
                    
                    ego_dists = [np.linalg.norm(np.array(pos) - intersection_point) 
                                for pos in ego_future_positions]
                    agent_dists = [np.linalg.norm(np.array(pos) - intersection_point) 
                                for pos in agent_future_positions]
                    
                    ego_time = ego_times[np.argmin(ego_dists)]
                    agent_time = agent_times[np.argmin(agent_dists)]
                    
                    # Check if vehicles are at intersection point within time threshold
                    if abs(ego_time - agent_time) < TIME_THRESHOLD:
                        collision_detected = True
                        # Find corresponding reference trajectory index
                        ref_traj_dists = [np.linalg.norm(np.array(pos) - intersection_point) 
                                        for pos in self.reference_trajectory]
                        conflict_idx = np.argmin(ref_traj_dists)
                        break  # Stop checking other points if collision is detected
            
            # Store results
            self.agent_collide.append(collision_detected)
            self.conflict_points.append(intersection_point if collision_detected else None)
            self.conflict_index.append(conflict_idx if collision_detected else None)

        # Check if there is any collision
        self.is_collide = np.any(self.agent_collide)
    
        # Update collision memory
        if self.is_collide:
            self.collision_memory = self.collision_memory_steps
        elif self.collision_memory > 0:
            self.collision_memory -= 1
            self.is_collide = True  # Keep treating it as collision for a few more steps

    def update_reference_states(self, speed_override=None, speed_overide_from_RL=None) -> np.ndarray:
        """Update reference states with improved safety checks and bounds."""
        DEFAULT_MAX_SPEED = 30.0
        SAFETY_BUFFER_POINTS = 5  # Number of points before conflict to reach zero speed
        
        # Initialize stop_point as None (will be used in plotting)
        self.stop_point = None
        
        # Handle RL speed override with bounds checking
        if speed_overide_from_RL is not None:
            new_ref = np.copy(self.reference_states)
            safe_speed = np.clip(speed_overide_from_RL[0,0], 0, DEFAULT_MAX_SPEED)
            new_ref[:, 2] = safe_speed
            return new_ref

        if not self.is_collide:
            return np.copy(self.reference_states)

        new_reference_states = np.copy(self.reference_states)
        
        # Get valid conflict indices (filter out None values)
        valid_conflict_indices = [idx for idx in self.conflict_index if idx is not None]
        
        if not valid_conflict_indices:
            return new_reference_states
            
        # Find earliest conflict point
        earliest_conflict_index = min(valid_conflict_indices)
        
        # Add safety buffer by moving the stopping point earlier
        stop_index = max(self.ego_index + 1, earliest_conflict_index - SAFETY_BUFFER_POINTS)
        
        # Calculate number of points between ego and stop point
        points_to_stop = stop_index - self.ego_index
        
        if points_to_stop > 0:
            # Create linear decrease from current speed to zero
            current_speed = self.ego_vehicle.speed
            deceleration_profile = np.linspace(current_speed, 0, points_to_stop)
            
            # Apply the deceleration profile up to the stop point
            new_reference_states[self.ego_index:stop_index, 2] = deceleration_profile
            
            # Set speed to zero from stop point onwards
            new_reference_states[stop_index:, 2] = 0.0
            
            # Store the stop point coordinates for plotting
            self.stop_point = self.reference_trajectory[stop_index]

        return new_reference_states

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

        # Plot reference trajectory in grey
        self.ax.scatter(
            x=self.reference_trajectory[:, 0], 
            y=self.reference_trajectory[:, 1], 
            color='grey', 
            s=1,
        )
        
        # Plot the ego vehicle's location in blue
        ego_x, ego_y = self.ego_vehicle.position
        self.ax.scatter(
            x=ego_x, 
            y=ego_y, 
            color='blue', 
            s=5
        )

        # Plot the agent vehicles' locations in red
        for agent_vehicle in self.agent_vehicles:
            agent_x, agent_y = agent_vehicle.position
            self.ax.scatter(
                x=agent_x, 
                y=agent_y, 
                color='red', 
                s=5
            )

        # Plot the potential collisions
        for current_loc, conflict_loc in zip(self.agent_current_locations, self.conflict_points):
            if conflict_loc is not None:
                xs = [current_loc[0], conflict_loc[0]]
                ys = [current_loc[1], conflict_loc[1]]
                self.ax.plot(xs, ys, 'x-', markersize=5)  # Use markersize for clarity

        # Plot the stop point if it exists
        if hasattr(self, 'stop_point') and self.stop_point is not None:
            self.ax.scatter(
                self.stop_point[0],
                self.stop_point[1],
                color='black',
                s=20,
                marker='s',
                # label='Stop point',
                zorder=5  # Make sure it's drawn on top
            )
            # Add a text annotation near the stop point
            self.ax.annotate(
                'STOP',
                (self.stop_point[0], self.stop_point[1]),
                xytext=(10, 10),
                textcoords='offset points',
                color='yellow',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.3)
            )

        # Add legend if stop point exists
        if hasattr(self, 'stop_point') and self.stop_point is not None:
            self.ax.legend()

        # Pause briefly to create animation effect
        plt.pause(0.1)