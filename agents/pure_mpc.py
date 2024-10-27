""" Pure MPC agent """

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env

from .base import Agent
from .utils import MPC_Action, Vehicle


class PureMPC_Agent(Agent):
    def __init__(
        self, 
        env: Env,
        cfg: dict, 
    ) -> None:
        """
        Initializer.
        
        Args:
            env: gym.Env, a highway-env gymnasium environment to retrieve the configuration.
            cfg: 
        """
        super().__init__(env, cfg)

        # Load weights for each cost component from the configuration
        self.default_weights = {
            key: self.config[f"weight_{key}"]
            for key in [
                "state", 
                "control", 
                "distance", 
                "collision", 
                "input_diff", 
                "final_state"
            ]
        }

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
        self.is_collision_detected, collision_points = self.check_potential_collision()
        ref = self.global_reference_states[:, :2]
        
        if weights_from_RL is not None:
            weights = self.default_weights
        else:
            weights = {
                key: weights_from_RL[i] 
                for i, key in enumerate([
                    "state", 
                    "control", 
                    "distance", 
                    "collision", 
                    "input_diff", 
                    "final_state"
                ])
            }
            
        weights = {
            key: weight for key, weight in zip([
                "state", 
                "control", 
                "distance", 
                "collision", 
                "input_diff", 
                "final_state"], 
                weights_from_RL)
            }

        closest_points = []
        if self.is_collision_detected:
            for point in collision_points:
                distances = np.linalg.norm(ref - point, axis=1) 
                closest_index = np.argmin(distances)
                closest_points.append((ref[closest_index], closest_index))
            self.collision_point_index = min(closest_points, key=lambda x: x[1])[1]
        else:
            self.collision_point_index = None
            
        mpc_action = self._solve(weights)
        return mpc_action.numpy() if return_numpy else mpc_action
      
    def _solve(self, weights: dict[str, float]) -> MPC_Action:
        
        # MPC parameters
        N = self.horizon
        
        # Define symbolic variables for states and controls
        n_states = 4    # [x, y, theta, v]
        n_controls = 2  # [acceleration, steer_angle]
        
        # Create symbolic variables for state and control trajectories
        x = ca.SX.sym('x', n_states, N + 1)  # State trajectory over N + 1 time steps
        u = ca.SX.sym('u', n_controls, N)    # Control inputs over N time steps
        
        ref = self.update_reference_states(index=self.collision_point_index)
        distances = [np.linalg.norm(self.ego_vehicle.position - np.array(point[:2])) for point in ref]
        closest_index = np.argmin(distances)

        # Define the cost function (objective to minimize)
        cost = 0
        state_cost = 0
        control_cost = 0
        final_state_cost = 0
        input_diff_cost = 0
        distance_cost = 0
        collision_cost = 0
        
        for k in range(N):
            ref_traj_index = min(closest_index + k, ref.shape[0] - 1)

            ref_v = ref[ref_traj_index,2]
            ref_heading = ref[ref_traj_index,3]
            
            """ CVXPY version """
            
            # d_perp = x[0, k] - ref[ref_traj_index,0]
            # d_para = x[1, k] - ref[ref_traj_index,1]
            
            # c_perp, s_perp = ca.cos(ref_heading + ca.pi/2), ca.sin(ref_heading + ca.pi/2)
            # matrix_perp = ca.SX([
            #     [c_perp**2, c_perp*s_perp],
            #     [c_perp*s_perp, s_perp**2]])
            
            # c_para, s_para = ca.cos(ref_heading), ca.sin(ref_heading)
            # matrix_para = ca.SX([
            #     [c_para**2, c_para*s_para],
            #     [c_para*s_para, s_para**2]])
            
            # total_d = ca.vertcat(d_perp, d_para)

            # # print(f"ref speed:{ref_v}, ref x: {ref[ref_traj_index,0]}, ref y: {ref[ref_traj_index,1]}")
            # perp_deviation = ca.norm_2(ca.mtimes(matrix_perp, total_d))
            # para_deviation = ca.norm_2(ca.mtimes(matrix_para, total_d))

            """ ChatGPT version"""

            # delta_x =  x[0, k] - ref[ref_traj_index, 0]
            # delta_y =  x[1, k] - ref[ref_traj_index, 1]

            # # Adjust for inverted y-axis if necessary
            # # delta_y *= -1  # If your coordinate system has y increasing downwards

            # ref_heading = ref[ref_traj_index, 3]
            # # print(ref_heading)

            # # Perpendicular Deviation (distance to the path)
            # perp_deviation = -ca.sin(ref_heading) * delta_x + ca.cos(ref_heading) * delta_y

            # # Parallel Deviation (progress along the path)
            # para_deviation = ca.cos(ref_heading) * delta_x + ca.sin(ref_heading) * delta_y

            """ Gozde version"""
            dx = x[0, k] - ref[ref_traj_index,0]
            dy = x[1, k] - ref[ref_traj_index,1]

            ref_v = ref[ref_traj_index,2]
            ref_heading = ref[ref_traj_index,3]
            perp_deviation = dx * ca.sin(ref_heading) - dy * ca.cos(ref_heading)
            para_deviation = dx * ca.cos(ref_heading) + dy * ca.sin(ref_heading)

            # State cost
            state_cost += (
                4 * perp_deviation**2 + 
                2 * para_deviation**2 +
                1 * (x[3, k] - ref_v)**2 + 
                0.1 * (x[2, k] - ref_heading)**2
            )

            # Control cost
            control_cost += 0.01 * u[0, k]**2 + 0.01 * u[1, k]**2
            
            # Input difference cost
            if k > 0:
                input_diff_cost += 0.01 * ((u[0, k] - u[0, k-1])**2 + (u[1, k] - u[1, k-1])**2)

            # Distance cost
            for other_vehicle in self.agent_vehicles:
                dist = ca.norm_2(x[:2, k] - other_vehicle.position)
                # in casadi, use ca.if_else to branch
                distance_cost += ca.if_else(
                    dist < 1.0, # if-statement
                    1000 / (dist + 1e-6)**2,  # if True 
                    100 / (dist + 1e-6)**2    # if False
                )

            # Collision cost
            collision_penalty_applied = 0   # flag: whether apply collision penalty on speed at this timestep

            for other_vehicle in self.agent_vehicles:
                # Get positions of the ego vehicle and the other vehicle
                ego_position = x[:2, k]  # [x_ego, y_ego]
                other_position = other_vehicle.position  # Assuming this is [x_other, y_other]
                
                # Calculate distance in x and y
                delta_x = ego_position[0] - other_position[0]
                delta_y = ego_position[1] - other_position[1]
                
                dist_to_other_vehicle_x = ca.fabs(delta_x)  # Distance in x direction
                dist_to_other_vehicle_y = ca.fabs(delta_y)  # Distance in y direction
                
                # Get speeds of the ego vehicle and the other vehicle
                ego_speed_x = x[3, k] * ca.cos(x[2, k])  # Assuming x[2, k] is the heading
                ego_speed_y = x[3, k] * ca.sin(x[2, k])
                
                other_speed_x = other_vehicle.speed * ca.cos(other_vehicle.heading)  # Assuming you have heading
                other_speed_y = other_vehicle.speed * ca.sin(other_vehicle.heading)
                
                # Calculate relative velocities in x and y
                relative_velocity_x = ego_speed_x - other_speed_x
                relative_velocity_y = ego_speed_y - other_speed_y
                
                # Calculate TTC for x and y directions using CasADi
                ttc_x = ca.if_else(relative_velocity_x < 0, dist_to_other_vehicle_x / (relative_velocity_x + 1e-6), ca.inf)
                ttc_y = ca.if_else(relative_velocity_y < 0, dist_to_other_vehicle_y / (relative_velocity_y + 1e-6), ca.inf)
                
                # Calculate collision cost based on TTC thresholds
                collision_condition = ca.logic_or(ttc_x < self.ttc_threshold, ttc_y < self.ttc_threshold) # Use ca.or_ for logical OR
                collision_penalty_applied = ca.logic_or(collision_condition, collision_penalty_applied) 

            # Apply penalty if not already applied
            collision_cost += ca.if_else(collision_penalty_applied, 3 * x[3, k]**2, 0)
            
            # Update other vehicles' location (constant speed)
            for other_vehicle in self.agent_vehicles:
                other_vehicle.position = self.other_vehicle_model(other_vehicle, self.dt)
        

        # final state cost
        ref_traj_index = min(closest_index + N, ref.shape[0] - 1)
        desired_final_state = ref[ref_traj_index, :]        
        final_state_cost += 100 * (
            (x[0, -1] - desired_final_state[0])**2 + 
            (x[1, -1] + desired_final_state[1])**2 + 
            (x[3, -1] - desired_final_state[2])**2 + # ref speed
            (x[2, -1] - desired_final_state[3])**2 # heading angle
        )

        cost = (
            state_cost * weights["weight_state"] + # old weight: 10
            control_cost * weights["weight_control"] + # old weight: 1
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
            'f': cost,          # objective (total cost to minimize)
            'g': g,             # constraint
        }
        
        opts = {
            'ipopt.print_level': 0, 
            'print_time':0,
            'ipopt.max_iter':1000,
            'ipopt.tol':1e-6,
        }
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Initial guess for the optimization
        state = np.array([self.ego_vehicle.position[0], self.ego_vehicle.position[1], self.ego_vehicle.heading, self.ego_vehicle.speed])
        x0_states = np.tile(state, (N + 1, 1)).flatten()  # Shape: (4 * (N + 1),)
        u0_controls = np.zeros(n_controls * N)           # Shape: (2 * N,)
        # Combine the initial guesses for states and controls
        x0 = np.concatenate((x0_states, u0_controls))    # Shape: (6 * N + 4,)
        
        # Solve the optimization problem
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        
        if not solver.stats()['success']:
            print("NOTICE: Not found solution")
            
        """ Print costs """
        # Extract the solution values
        opt_values = sol['x'].full()
        u_opt = opt_values[n_states * (N + 1):].reshape((N, n_controls))
        
        # Calculate and print the state cost and other cost components
        state_cost_val, control_cost_val, final_state_cost_val, input_diff_cost_val, distance_cost_val, collision_cost_val = cost_fn(opt_values)
        
        # print("\n------")
        # print(f"State Cost: {state_cost_val}")
        # print(f"Control Cost: {control_cost_val}")
        # print(f"Final State Cost: {final_state_cost_val}")
        # print(f"Input Difference Cost: {input_diff_cost_val}")
        # print(f"Distance Cost: {distance_cost_val}")
        # print(f"Collision Cost: {collision_cost_val}")

        u_opt = sol['x'][n_states * (N + 1):].full().reshape((N, n_controls))
        
        mpc_action = MPC_Action(acceleration=u_opt[0, 0], steer=u_opt[0, 1])
        return mpc_action
    
    def plot(self, width: float = 100, height: float = 100):
        plt.clf() 
        plt.scatter(self.global_reference_states[:,0], self.global_reference_states[:,1], c='grey', s=1)
        plt.scatter(self.ego_vehicle.position[0], self.ego_vehicle.position[1])
        plt.axis('on')  
        plt.xlim([-width, width])
        plt.ylim([-height, height])

        plt.gca().invert_yaxis() # flip y-axis, consistent with pygame window
        
        plt.pause(0.1) # animate