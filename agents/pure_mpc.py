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
            cfg: 
        """
        super().__init__(env, cfg)

        if cfg["render"]:
            self.fig, self.ax = plt.subplots(figsize=(4, 4))

        # Load weights for each cost component from the configuration
        self.default_weights = {
            f"weight_{key}": self.config[f"weight_{key}"]
            for key in PureMPC_Agent.weight_components
        }

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
        self.check_collision()
        mpc_action = self._solve(weights_from_RL)
        return mpc_action.numpy() if return_numpy else mpc_action
      
    def _solve(self, weights_from_RL: dict[str, float]) -> MPC_Action:
        
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

        """ Update ref speed """
        self.is_collision_detected, collision_points = self.check_potential_collision()
        ref = self.global_reference_states[:, :2]

        distances = [np.linalg.norm(self.ego_vehicle.position - np.array(point[:2])) for point in ref]
        self.ego_index = np.argmin(distances)

        closest_points = []
        if self.is_collision_detected:
            for point in collision_points:
                distances = np.linalg.norm(ref - point, axis=1) 
                closest_index = np.argmin(distances)
                closest_points.append((ref[closest_index], closest_index))
            self.collision_point_index = min(closest_points, key=lambda x: x[1])[1]
        else:
            self.collision_point_index = None
        
        ref = self.update_reference_states(
            ego_index=self.ego_index,
            conflict_index=self.collision_point_index,
            speed_override=10)
        
        # distances = [np.linalg.norm(self.ego_vehicle.position - np.array(point[:2])) for point in ref]
        closest_index = self.ego_index

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
        self.last_acc = u_opt[0, 0]

        mpc_action = MPC_Action(acceleration=u_opt[0, 0], steer=u_opt[0, 1])
        return mpc_action
    
    def plot(self, width: float = 30, height: float = 30):
        self.ax.clear()
        
        # Set limits and invert y-axis once per plot call
        self.ax.set_xlim([-width, width])
        self.ax.set_ylim([-height, height])
        self.ax.invert_yaxis()
        self.ax.grid(True)
        
        # Scatter and plot on the existing ax
        self.ax.scatter(self.global_reference_states[:, 0], self.global_reference_states[:, 1], c='grey', s=1)
        self.ax.scatter(self.ego_vehicle.position[0], self.ego_vehicle.position[1], c='blue', s=5)

        for agent, point in zip(self.agent_vehicles, self.conflict_points):
            # self.ax.scatter(agent.position[0], agent.position[1], c='red', s=5)
            if point is not None:
                xs, ys = [agent.position[0], point[0]], [agent.position[1], point[1]]
                self.ax.plot(xs, ys, 'x-')

        # for start_point, end_point in self.points:
        #     xs = (start_point[0], end_point[0])
        #     ys = (start_point[1], end_point[1])
        #     plt.plot(xs, ys, "-")
        #     plt.scatter(xs[0], ys[0], marker='o')
        #     plt.scatter(xs[-1], ys[-1], marker='s')

        # Pause briefly to create animation effect
        plt.pause(0.1) 


    def check_collision(self) -> bool:
        """ Check the potential collison between ego vehicle and agent vehicles in future """

        segment_length = 50  

        self.points = list()
        agent_props = list()
        for agent in self.agent_vehicles:
            agent_location = agent.position
            agent_speed = agent.speed
            agent_heading_angle = agent.heading

            # heading_rad = np.radians(agent_heading_angle)
            delta_x = segment_length * np.cos(agent_heading_angle)
            delta_y = segment_length * np.sin(agent_heading_angle)

            start_point = agent_location
            end_point = agent_location + np.array([delta_x, delta_y])
            agent_segment = LineString([start_point, end_point])

            agent_props.append((agent_location, agent_speed, agent_heading_angle, agent_segment))
            self.points.append((start_point, end_point))

        ego_path_lineString = LineString(self.reference_states[:, :2])
        
        
        self.conflict_points = list()

        for agent_prop in agent_props:
            agent_lineString = agent_prop[3]
            intersection = ego_path_lineString.intersection(agent_lineString)

            if not intersection.is_empty:
                if intersection.geom_type == 'Point':
                    # print("Point:", type(intersection))
                    self.conflict_points.append((intersection.x, intersection.y))
            elif intersection.geom_type == 'MultiPoint':
                print("MultiPoint:", intersection)
                points = []
                for point in intersection.geoms:
                    points.append(point.x, point.y)
                self.conflict_points.append(tuple(points))

            else:
                self.conflict_points.append(None)

        agent_distances = list()
        for agent_id, conflict_point in enumerate(self.conflict_points):
            if conflict_point is not None:
                agent_distances.append(
                    np.linalg.norm(
                        agent_props[agent_id][0] - np.array(conflict_point[:2])
                    )
                )
            else:
                agent_distances.append(None)

        agent_ETAs = list()
        for agent_id, distance in enumerate(agent_distances):
            if distance is not None:
                agent_ETAs.append(distance/agent_props[agent_id][1])
            else:
                agent_ETAs.append(-1*100)

        distances = [np.linalg.norm(self.ego_vehicle.position - np.array(point[:2])) for point in self.reference_states]
        self.ego_index = np.argmin(distances)

        ego_ETAs = list()
        coords = list(ego_path_lineString.coords)
        for agent_id, conflict_point in enumerate(self.conflict_points):
            if conflict_point is not None:
                distances = [np.linalg.norm(conflict_point[:2] - np.array(point[:2])) for point in self.reference_states]
                agent_index = np.argmin(distances)
                distance = LineString(self.reference_states[self.ego_index: agent_index,:2]).length
                ego_ETAs.append(self.calculate_ego_eta(
                    distance, acceleration=self.last_acc,
                ))

            else:
                ego_ETAs.append(100)

        delta_time = np.nan_to_num(ego_ETAs, nan=np.inf) - np.nan_to_num(agent_ETAs, nan=-1*np.inf)
        return np.any(delta_time < 1)


    def calculate_ego_eta(self, distance, acceleration: float = 1, max_speed=10):
        current_speed = self.ego_vehicle.speed
        
        if current_speed >= max_speed:
            # already reach speed limit
            return distance / max_speed
        else:
            # need to accelerate first
            distance_when_vmax = (max_speed**2 - current_speed**2) / acceleration / 2
            if distance > distance_when_vmax:
                return (max_speed - current_speed) / acceleration + (distance - distance_when_vmax) / max_speed
            else:
                v_max = np.sqrt(2*acceleration*distance + current_speed**2)
                return (v_max - current_speed) / acceleration