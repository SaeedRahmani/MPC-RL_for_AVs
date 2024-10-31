""" Pure MPC agent """

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env
from shapely import LineString

from .base import Agent
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

        # Load collision detection parameters from config
        self.detection_dist = self.config.get("detection_distance", 100)
        self.ttc_threshold = self.config.get("ttc_threshold", 1)

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
        mpc_action = self._solve(weights_from_RL)
        return mpc_action.numpy() if return_numpy else mpc_action
      
    def _solve(self, weights_from_RL: dict[str, float]=None) -> MPC_Action:
        
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
                f"weight_{key}": weights_from_RL[i] 
                for i, key in enumerate(PureMPC_Agent.weight_components)
            }

        # Get the index on the reference trajectory for ego vehicle
        self.ego_index = np.argmin(
            [np.linalg.norm(self.ego_vehicle.position - trajectory_point) 
             for trajectory_point in self.reference_trajectory]
        )
        
        # Generate new reference states, given the result of collision detection
        ref = self.update_reference_states(speed_override=self.config["speed_override"])
        
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
                print('collision', self.is_collide)
                print('ref v', ref_v)
                speed_weight = 200
                
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

        print('lbx', lbx)
        print('ubx', ubx)
        
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
        print('optimal values', opt_values)
        u_opt = sol['x'][-N * n_controls:].full().reshape(N, n_controls)
        print('optimal input values', u_opt)
        print('first acc', u_opt[0, 0])

        self.last_acc = u_opt[0, 0]

        mpc_action = MPC_Action(acceleration=u_opt[0, 0], steer=u_opt[0, 1])
        return mpc_action
    
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
        # lineplot with x-markers at each end
        for current_loc, conflict_loc in zip(self.agent_current_locations, self.conflict_points):
            if conflict_loc is not None:
                xs = [current_loc[0], conflict_loc[0]]
                ys = [current_loc[1], conflict_loc[1]]
                self.ax.plot(xs, ys, 'x-', markersize=5)  # Use markersize for clarity

        # Pause briefly to create animation effect
        plt.pause(0.1)

    def _check_collision(self):
        """ Collision detection """

        # Create a LineString for ego reference trajectory
        ego_path_lineString = LineString(self.reference_trajectory)

        # Get the index of closest point on the reference trajectory towards the ego vehicle
        ego_location = np.array(self.ego_vehicle.position)
        ego_index = np.argmin([
            np.linalg.norm(ego_location - np.array(trajectory_point)) 
            for trajectory_point in self.reference_trajectory
        ])
        self.ego_index = ego_index

        # Use self. for visualization (self.plot)
        self.agent_current_locations = list()
        self.agent_future_locations = list()
        self.conflict_points = list()
        self.conflict_index = list()
        self.agent_collide = list()

        for _, agent_veh in enumerate(self.agent_vehicles):
            # store the current location of agent vehicle
            agent_current_location = np.array(agent_veh.position)
            self.agent_current_locations.append(agent_current_location)

            # store the future location of agent vehicle
            agent_future_location = agent_current_location + self.detection_dist * np.array([
                    np.cos(agent_veh.heading), np.sin(agent_veh.heading)])
            self.agent_future_locations.append(agent_future_location)
            
            # create a LineString object for the path between current and future locations
            agent_path_lineString = LineString([agent_current_location, agent_future_location])

            # check whether there is an intersection
            intersection = ego_path_lineString.intersection(agent_path_lineString)
            if not intersection.is_empty:
                if intersection.geom_type == 'Point':
                    conflict_location = np.array((intersection.x, intersection.y))
                    agent_index = np.argmin([np.linalg.norm(conflict_location - point)
                        for point in self.reference_trajectory])
                    if ego_index < agent_index:
                        agent_dist = np.linalg.norm(agent_current_location - conflict_location)
                        agent_speed = agent_veh.speed
                        agent_eta = agent_dist / agent_speed

                        ego_dist = LineString(self.reference_trajectory[ego_index: agent_index+1,:]).length
                        # ego_speed = self.ego_vehicle.speed
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
        # return self.is_collide

    ### This new function might be correct but needs revision
    def calculate_ego_eta(self, distance, ego_index, acceleration: float = 1, max_speed=10):
        current_speed = self.ego_vehicle.speed
        # acceleration = self.last_acc
        acceleration = 0
        print('current speed', current_speed)
        print('current acceleration', acceleration)
        # Ego vehicle's velocity direction
        velocity_vector = np.array([
            current_speed * np.cos(self.ego_vehicle.heading),
            current_speed * np.sin(self.ego_vehicle.heading),
        ])

        # Reference trajectory direction at the target point
        trajectory_vector = np.array([
            max_speed * np.cos(self.reference_states[ego_index, 3]),
            max_speed * np.sin(self.reference_states[ego_index, 3]),
        ])

        # Determine if the vehicle is moving along the trajectory direction
        sign = 1 if np.dot(velocity_vector, trajectory_vector) > 0 else -1
        adjusted_speed = sign * current_speed

        if acceleration == 0:
            if adjusted_speed == 0:
                return np.inf  # Can't reach the point without moving
            else:
                return distance / adjusted_speed

        # Solve quadratic equation 0.5 * a * t^2 + v0 * t - distance = 0
        a = 0.5 * acceleration
        b = adjusted_speed
        c = -distance

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return np.inf  # Can't reach the distance

        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b + sqrt_discriminant) / (2 * a)
        t2 = (-b - sqrt_discriminant) / (2 * a)

        # Choose the positive time
        t = max(t1, t2)

        if t < 0:
            return np.inf

        return t
    
    def update_reference_states(self, speed_override=None) -> np.ndarray:
        if not self.is_collide:
            return np.copy(self.reference_states)
        else:
            new_reference_states = np.copy(self.reference_states)
            earliest_conflict_index = np.min(self.conflict_index)
            n_points = earliest_conflict_index - self.ego_index
            distance_to_conflict = LineString(self.reference_trajectory[self.ego_index: earliest_conflict_index+1, :2]).length
            distance_to_conflict = distance_to_conflict - 2 # buffer
            # Calculate required deceleration
            current_speed = self.ego_vehicle.speed
            max_deceleration = - self.ego_vehicle.max_deceleration  # Should be positive
            required_deceleration = - (current_speed ** 2) / (2 * distance_to_conflict)
            required_deceleration = max(required_deceleration, -max_deceleration)
            
            # Calculate speeds at each point
            for i in range(self.ego_index, earliest_conflict_index):
                trajectory_slice = self.reference_trajectory[self.ego_index: i+1, :2]
                if len(trajectory_slice) > 1:
                    distance = LineString(trajectory_slice).length
                else:
                    distance = 0  # or handle this case as appropriate

                speed = np.sqrt(max(current_speed ** 2 + 2 * required_deceleration * distance, 0))
                new_reference_states[i, 2] = speed
            
            # Zero out ref speed after conflict point
            new_reference_states[earliest_conflict_index:, 2] = 0
            
            return new_reference_states
      

    ''' Function from Gozde's implementation -- to be checked and used for collision avoidance'''  
    # def calculate_distances(current_state, obstacles):
    #     current_position = current_state[:2]
    #     distances = [np.linalg.norm(current_position - np.array(obs[:2])) for obs in obstacles]
    #     return distances
    
    # def determine_direction(ego_psi, other_psi):
    #     # Calculate the absolute difference in heading
    #     angle_diff = abs(ego_psi - other_psi)
    #     # Normalize the difference to the range [0, pi]
    #     angle_diff = angle_diff % np.pi
    #     # Determine direction based on a threshold (e.g., 45 degrees in radians)
    #     threshold = np.pi / 4
    #     if angle_diff < threshold or angle_diff > (np.pi - threshold):
    #         return "same"
    #     else:
    #         return "opposite"
    
    # def vehicle_model(state, action):
    #     x, y, v, psi = state
    #     a, delta = action
        
    #     beta = math.atan(0.5 * math.tan(delta))
    #     x_next = x + v * math.cos(psi + beta) * dt
    #     y_next = y + v * math.sin(psi + beta) * dt
    #     v_next = max(0, v + a * dt)  # Ensure that speed does not go negative
    #     psi_next = psi + v * math.sin(beta) / WHEELBASE * dt
        
    #     return np.array([x_next, y_next, v_next, psi_next])

    # def find_closest_point(current_state, reference_trajectory):
    #     current_position = current_state[:2]
    #     distances = [np.linalg.norm(current_position - np.array(point[:2])) for point in reference_trajectory]
    #     closest_index = np.argmin(distances)
    #     return closest_index
    
    # def predict_vehicle_positions(state, action, steps, dt=0.1):
    #     """ Predict future positions of a vehicle based on current state and action """
    #     state = np.array(state)
    #     predictions = []

    #     for _ in range(steps):
    #         state = vehicle_model(state, action)
    #         predictions.append(state[:2])  # Append only x, y positions
        
    #     #print("predicted obstacles-----",predictions)
    #     return predictions
        
    # def predict_others_future_positions(obstacles, ego_speed, steps, dt):
    #     """ Predict future positions of all obstacles """
    #     future_positions = []
        
    #     for obstacle in obstacles:
    #         if len(obstacle) == 2:
    #             x, y = obstacle
    #             vx, vy = 0, 0  # Default to zero velocity if not provided
    #             psi = 0
    #         else:
    #             x, y, vx, vy, psi, direction = obstacle
    #             if direction == "same":
    #                 # Add ego vehicle speed to the obstacle's speed
    #                 v = np.sqrt(vx**2 + vy**2) + ego_speed
    #                 # Recompute vx and vy based on the new speed
    #                 vx = v * np.cos(psi)
    #                 vy = v * np.sin(psi)
    #             else:
    #                 v = np.sqrt(vx**2 + vy**2)

    #         state = [x, y, v, psi]
            
    #         action = [0, 0]  # Assuming constant velocity model for simplicity
    #         future_positions.append(predict_vehicle_positions(state, action, steps, dt))
        
        
    #     return future_positions

    # def check_collisions(predicted_ego_path, predicted_obstacles, start_index=0):
    #     collision_points = []
    #     collision_detected = False
    #     buffer = 2.6
    #     half_width = buffer / 2 
    #     time_steps_window = int(1.0 / dt)  # Number of steps to check within the time window
        
    #     for step, (px, py) in enumerate(predicted_ego_path[start_index:]):
    #         ego_box = [(px - half_width, py - half_width), (px + half_width, py + half_width)]
    #         for obs_future_positions in predicted_obstacles:
    #             for obs_step in range(max(0, step - time_steps_window), min(len(obs_future_positions), step + time_steps_window + 1)):
    #                 ox, oy = obs_future_positions[obs_step]
    #                 distance_to_obstacle = np.sqrt((px - ox)**2 + (py - oy)**2)
    #                 if distance_to_obstacle < 1.0:
    #                     if (ego_box[0][0] <= ox <= ego_box[1][0]) and (ego_box[0][1] <= oy <= ego_box[1][1]):
    #                         collision_points.append((px, py))
    #                         collision_detected = True
    #                         break  # Stop checking further obstacles if collision is detected
    #     return collision_points, collision_detected
    
    # def process_observation(obs):
    #     ego_vehicle = obs[0]
        
    #     x, y = ego_vehicle[1], ego_vehicle[2]
    #     vx, vy = ego_vehicle[3], ego_vehicle[4]
    #     v = np.sqrt(vx**2 + vy**2)
        
    #     psi = np.arctan2(vy, vx)  # Calculate heading from velocity
        
    #     current_state = np.array([x, y, v, psi])
        
    #     obstacles = []
    #     directions = []
    #     for vehicle in obs[1:]:
    #         if vehicle[0] == 1:
    #             ox, oy = vehicle[1], vehicle[2]
    #             ovx = vehicle[3] if len(vehicle) > 3 else 0  # Set default velocity components if not available
    #             ovy = vehicle[4] if len(vehicle) > 4 else 0
    #             o_v = np.sqrt(ovx**2 + ovy**2)
    #             o_psi = np.arctan2(ovy, ovx)
    #             direction = determine_direction(psi, o_psi)
    #             obstacles.append([ox, oy, ovx, ovy, o_psi, direction])
    #             directions.append(direction)
        
    #     return current_state, obstacles, directions
