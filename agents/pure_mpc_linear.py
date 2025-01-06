"""
Iterative Linear MPC Agent, adapted to your existing base Agent class,
which already provides:
 - reference_states => Nx4 array: (x, y, v, heading)
 - self.ego_vehicle, self.agent_vehicles after _parse_obs()
 - A built-in "render" config flag
 - A built-in reference_trajectory (via self.reference_trajectory)
And we add a 'plot()' method just like your old code, so you can visualize
the reference trajectory and vehicles.

Dependencies:
    pip install cvxpy matplotlib
"""

import math
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from .base_agent import Agent
from .utils import MPC_Action

NX = 4  # state = (x, y, v, heading)
NU = 2  # input = (acceleration, steering)

# Cost weighting:
R = np.diag([0.01, 0.01])     # input cost
Rd = np.diag([0.01, 1.0])     # input difference cost
Q_v_yaw = np.diag([20, 0.5])  # cost on [v, heading] deviation
Qf = np.diag([1.0, 1.0, 0.0, 0.5])  # final-state cost (scaled by horizon)

# Bounds
MAX_STEER = math.radians(30.0)   # max steering angle
MAX_DSTEER = math.radians(30.0)  # max steering rate (rad/s)
MAX_ACCEL = 2.0                  # [m/s^2]
MAX_DECEL = -5.0                 # [m/s^2]
MAX_SPEED = 40 / 3.6           # [m/s] ~ 6.94 m/s

def calc_nearest_index_in_direction(px, py, path_x, path_y, start_idx=0):
    """
    Finds the index on (path_x, path_y) that is closest to (px, py),
    searching forward from start_idx.
    """
    if start_idx >= len(path_x):
        return len(path_x) - 1

    min_dist = float("inf")
    best_idx = start_idx
    for i in range(start_idx, len(path_x)):
        dx = path_x[i] - px
        dy = path_y[i] - py
        dist = dx*dx + dy*dy
        if dist < min_dist:
            min_dist = dist
            best_idx = i
        else:
            # optional: early break if distance starts to increase
            pass

    return best_idx

def linear_model_matrix(v_bar, yaw_bar, steer_ref, dt, wheelbase):
    """
    Build linear system matrices A, B, C around the operating point.
    Kinematic bicycle model with discretization for one time step dt.
    """
    A = np.eye(NX)
    A[0, 2] = dt * math.cos(yaw_bar)
    A[0, 3] = -dt * v_bar * math.sin(yaw_bar)
    A[1, 2] = dt * math.sin(yaw_bar)
    A[1, 3] =  dt * v_bar * math.cos(yaw_bar)
    A[3, 2] =  dt * math.tan(steer_ref) / wheelbase  # partial derivative wrt v

    B = np.zeros((NX, NU))
    B[2, 0] = dt  # accel -> speed
    denom = wheelbase * (math.cos(steer_ref) ** 2)
    if abs(denom) < 1e-12:
        denom = 1e-12
    B[3, 1] = dt * v_bar / denom  # steering -> heading

    C = np.zeros(NX)
    return A, B, C

def predict_motion(x0, accel_profile, steer_profile, dt, wheelbase):
    """
    Forward-simulate a nominal (operational) trajectory from x0,
    applying the control sequences (accel_profile, steer_profile).
    Returns xbar shape = (NX, T+1).
    """
    T_ = len(accel_profile)
    xbar = np.zeros((NX, T_ + 1))
    xbar[:, 0] = x0

    cur_x, cur_y, cur_v, cur_yaw = x0
    for i in range(T_):
        a_i = accel_profile[i]
        d_i = steer_profile[i]
        # kinematic update
        cur_v += a_i * dt
        # clamp speed
        cur_v = max(0.0, min(cur_v, MAX_SPEED))
        cur_yaw += (cur_v / wheelbase) * math.tan(d_i) * dt
        cur_x += cur_v * math.cos(cur_yaw) * dt
        cur_y += cur_v * math.sin(cur_yaw) * dt

        xbar[0, i+1] = cur_x
        xbar[1, i+1] = cur_y
        xbar[2, i+1] = cur_v
        xbar[3, i+1] = cur_yaw
    return xbar

class IterativeLinearMPC_Agent(Agent):
    """
    Iterative linear MPC agent, leveraging your base Agent for:
     - reference_states => shape (N,4): (x, y, v, heading)
     - horizon => T
     - dt => sampling time
     - self.ego_vehicle => parse obs
     - self.agent_vehicles => parse obs
    Also includes a plot() method that shows the reference trajectory,
    ego vehicle, and agent vehicles, just like your old code.
    """

    def __init__(self, env, cfg):
        super().__init__(env, cfg)

        # We'll store old solutions for warm starts
        self.oa = None  # acceleration profile
        self.od = None  # steering profile

        # For the linear model, we need a wheelbase or length parameter
        # (from config or a default guess):
        self.wheelbase = self.config.get("wheelbase", 2.5)

        # Possibly scale Qf by T to replicate your old code
        self.Qf_mod = Qf * self.horizon

        # If rendering is enabled, create a figure+ax for plotting
        if self.render:
            window_size = self.config.get("render_window_size", 5)
            self.fig, self.ax = plt.subplots(figsize=(window_size, window_size))
            manager = plt.get_current_fig_manager()
            # If you're on certain IDEs or notebooks, manager might be None
            # so guard it:
            if manager is not None:
                manager.window.wm_geometry("+50+50")

    def __str__(self):
        return "Iterative Linear MPC Agent"

    def _solve(self) -> MPC_Action:
        """
        Solve the iterative linear MPC, returning one-step control (acc, steer).
        We'll do one iteration of linearization, like your old snippet.
        """
        T_ = self.horizon
        # reference_states => shape (N,4) => columns [x, y, v, heading]
        ref_states = self.global_reference_states

        # Ego state from obs:
        ex = self.ego_vehicle.position[0]
        ey = self.ego_vehicle.position[1]
        ev = self.ego_vehicle.speed
        eyaw = self.ego_vehicle.heading

        # Find nearest index on the reference
        path_x = ref_states[:, 0]
        path_y = ref_states[:, 1]
        self.target_ind = calc_nearest_index_in_direction(ex, ey, path_x, path_y, 0)
        # Build local reference for T+1 steps
        xref = np.zeros((NX, T_ + 1))
        for i in range(T_ + 1):
            idx = self.target_ind + i
            if idx >= len(ref_states):
                idx = len(ref_states) - 1
            xref[0, i] = ref_states[idx, 0]  # ref x
            xref[1, i] = ref_states[idx, 1]  # ref y
            xref[2, i] = ref_states[idx, 2]  # ref speed
            xref[3, i] = ref_states[idx, 3]  # ref heading

        # Warm start
        if self.oa is None or self.od is None:
            self.oa = np.zeros(T_)
            self.od = np.zeros(T_)

        x0 = np.array([ex, ey, ev, eyaw], dtype=float)

        # We'll do 1 iteration
        for _ in range(1):
            # forward-simulate to get xbar
            xbar = predict_motion(x0, self.oa, self.od, self.dt, self.wheelbase)
            # solve the linear MPC around xbar
            oa_new, od_new = self._linear_mpc_control(xref, xbar, x0)
            if oa_new is None:
                # solver failed => fallback
                return MPC_Action(0.0, 0.0)
            self.oa = oa_new
            self.od = od_new

        # Return first control
        acc_cmd = self.oa[0]
        steer_cmd = self.od[0]
        return MPC_Action(acc_cmd, steer_cmd)

    def _linear_mpc_control(self, xref, xbar, x0):
        """
        Solve the linearized MPC with CVXPY.
        Returns (oa, od) if feasible, else (None, None).
        """
        T_ = self.horizon
        dt_ = self.dt
        L_ = self.wheelbase

        x = cp.Variable((NX, T_ + 1))
        u = cp.Variable((NU, T_))

        cost = 0.0
        constraints = []
        constraints += [x[:, 0] == x0]

        for t in range(T_):
            # operating point
            v_bar = xbar[2, t]
            yaw_bar = xbar[3, t]
            steer_ref = 0.0  # or xref[3, t] if you want 'reference steer'

            A, B, C = linear_model_matrix(v_bar, yaw_bar, steer_ref, dt_, L_)

            # x_{t+1} = A x_t + B u_t + C
            constraints += [x[:, t+1] == A@x[:, t] + B@u[:, t] + C]

            # input cost
            cost += cp.quad_form(u[:, t], R)
            # input difference cost
            if t < T_ - 1:
                cost += cp.quad_form(u[:, t+1] - u[:, t], Rd)
                # steering rate limit
                constraints += [
                    cp.abs(u[1, t+1] - u[1, t]) <= MAX_DSTEER * dt_
                ]

            # penalize v,yaw deviation from reference
            cost += cp.quad_form(x[2:4, t] - xref[2:4, t], Q_v_yaw)

        # final-state cost
        cost += cp.quad_form(x[:, T_] - xref[:, T_], self.Qf_mod)

        # bounds
        for t in range(T_):
            constraints += [
                u[0, t] <= MAX_ACCEL,
                u[0, t] >= MAX_DECEL,
                cp.abs(u[1, t]) <= MAX_STEER
            ]
        for t in range(T_+1):
            constraints += [
                x[2, t] >= 0.0,     # no negative speed
                x[2, t] <= MAX_SPEED
            ]

        problem = cp.Problem(cp.Minimize(cost), constraints)
        result = problem.solve(solver=cp.ECOS, verbose=False)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print("[IterativeLinearMPC] MPC solve failed: status=", problem.status)
            return None, None

        oa = u.value[0, :]
        od = u.value[1, :]

        return oa, od

    def plot(self):
        """
        Visualize the MPC solving process, referencing your old code style.
        If 'render' is false, this won't do anything. 
        """
        if not self.render:
            return

        x_range = self.config.get("render_axis_range", 50)

        # Clear the old frame
        self.ax.clear()

        # Set limits and invert y-axis if your env uses inverted Y
        # or remove invert_yaxis() if you don't want that behavior
        self.ax.set_xlim([-x_range, x_range])
        self.ax.set_ylim([-x_range, x_range])
        self.ax.invert_yaxis()
        self.ax.grid(True)

        # Labels
        self.ax.set_xlabel("X (meter)")
        self.ax.set_ylabel("Y (meter)")
        self.ax.set_title("Iterative Linear MPC Agent")

        # Plot reference trajectory in grey
        # (self.reference_trajectory is (N,2) => [x, y] from base agent)
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

        # Pause briefly to create animation effect
        plt.pause(0.1)
