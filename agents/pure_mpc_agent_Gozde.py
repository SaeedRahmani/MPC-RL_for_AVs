""" Pure MPC agent """

import numpy as np
import matplotlib.pyplot as plt
from gymnasium import Env
import casadi as ca

from .agent import Agent
from .utils import MPC_Action, Vehicle

class PureMPC_Agent(Agent):
    def __init__(
        self, 
        env: Env, 
        horizon: int = 16, 
        render: bool = True
    ) -> None:
        """
        Initializer.
        
        Args:
            env: gym.Env, a highway-env gymnasium environment to retrieve the configuration.
            horizon: int, time horizon parameter in MPC.
            render: bool, whether display the mpc prediction process in a seperate animation.
        """
        super().__init__(env, horizon, render)
    
    def __str__(self) -> str:
        return "Pure MPC agent [Receding Horizon Control], Solved by `CasADi` "
      
    def _solve(self) -> MPC_Action:
        
        # MPC parameters
        N = self.horizon
        
        # Define symbolic variables for states and controls
        n_states = 4    # [x, y, theta, v]
        n_controls = 2  # [acceleration, steer_angle]
        
        # Create symbolic variables for state and control trajectories
        x = ca.SX.sym('x', n_states, N + 1)  # State trajectory over N + 1 time steps
        u = ca.SX.sym('u', n_controls, N)    # Control inputs over N time steps
        
        # TODO: Reference trajectory
        ref = self.reference_states
        distances = [np.linalg.norm(self.ego_vehicle.position - np.array(point[:2])) for point in self.reference_states]
        closest_index = np.argmin(distances)

        # Define the cost function (objective to minimize)
        cost = 0
        state_cost = 0
        control_cost = 0
        final_state_cost = 0
        input_diff_cost = 0
        
        for k in range(N):
            ref_traj_index = min(closest_index + k, self.reference_states.shape[0] - 1)

            dx = x[0, k] - ref[ref_traj_index,0]
            dy = x[1, k] - ref[ref_traj_index,1]

            ref_v = ref[ref_traj_index,2]
            ref_heading = ref[ref_traj_index,3]
            perp_deviation = dx * ca.sin(ref_heading) - dy * ca.cos(ref_heading)
            para_deviation = dx * ca.cos(ref_heading) + dy * ca.sin(ref_heading)

            # State cost
            state_cost += (
                2 * perp_deviation**2 + 
                2 * para_deviation**2 +
                1 * (x[3, k] - ref_v)**2 + 
                1 * (x[2, k] - ref_heading)**2
            )

            # Control cost
            control_cost += 0.01 * u[0, k]**2 + 0.1 * u[1, k]**2
            
            # Input difference cost
            if k > 0:
                input_diff_cost += 0.5 * ((u[0, k] - u[0, k-1])**2 + (u[1, k] - u[1, k-1])**2)

        # final state cost
        ref_traj_index = min(closest_index + N, self.reference_states.shape[0] - 1)
        desired_final_state = self.reference_states[ref_traj_index, :]        
        final_state_cost += 100 * (
            (x[0, -1] - desired_final_state[0])**2 + 
            (x[1, -1] + desired_final_state[1])**2 + 
            (x[3, -1] - desired_final_state[2])**2 + # ref speed
            (x[2, -1] - desired_final_state[3])**2 # heading angle
        )

        cost = 5 * state_cost + 20 * control_cost + final_state_cost + input_diff_cost

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
        nlp = {'x': opt_variables, 'f': cost, 'g': g}
        opts = {
            'ipopt.print_level':0, 
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
            
        u_opt = sol['x'][n_states * (N + 1):].full().reshape((N, n_controls))
        
        mpc_action = MPC_Action(acceleration=u_opt[0, 0], steer=u_opt[0, 1])
        return mpc_action
    
    def plot(self, width: float = 100, height: float = 100):
        plt.clf() 
        plt.scatter(self.reference_states[:,0], self.reference_states[:,1], c='grey', s=1)
        plt.scatter(self.ego_vehicle.position[0], self.ego_vehicle.position[1])
        plt.axis('on')  
        plt.xlim([-width, width])
        plt.ylim([-height, height])

        plt.gca().invert_yaxis() # flip y-axis, consistent with pygame window
        
        plt.pause(0.1) # animate
