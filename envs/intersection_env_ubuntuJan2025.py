from typing import Dict, Text, Tuple

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle


class IntersectionEnv(AbstractEnv):
    ACTIONS: Dict[int, str] = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": False,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": False,
                    "target_speeds": [0, 4.5, 9],  # Keep aligned with target_speed below
                },
                "reward_speed_range": [7.0, 9.0],  # Add this line
                "reward_weights": {
                "collision": -30.0,    # Larger penalty to offset higher speed reward
                "arrival": 25.0,       # Increased to maintain goal priority
                "speed": 2.0,          # 4x higher than previous
                "proximity_penalty": -0.5,  # Stronger penalty for safety
                "on_road": 2.0         # Emphasize staying on road
                },
                "target_speed": 9.0,
                "min_safe_distance": 2.0,
                "max_speed": 9.0,
                "duration": 10e6,            # [s] Maximum episode length
                "destination": "o1",
                "controlled_vehicles": 1,
                "initial_vehicle_count": 10,
                "spawn_probability": 0.6,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "scaling": 5.5 * 1.3,
                "normalize_reward": False,
                "offroad_terminal": False,  # Terminate episode if off-road
            }
        )
        return config

    # Reward only for ego vehicle
    def _reward(self, action: int) -> float:
        """Reward for ego vehicle only."""
        # Get the ego vehicle (first controlled vehicle)
        ego_vehicle = self.controlled_vehicles[0]
        
        # Return reward only for ego vehicle
        return self._agent_reward(action, ego_vehicle)

    # ORIGINAL ONE - Reward for all agents
    # def _reward(self, action: int) -> float:
    #     """Aggregated reward, for cooperative agents."""
    #     return sum(
    #         self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
    #     ) / len(self.controlled_vehicles)

    def _rewards(self, action: int) -> Dict[Text, float]:
        """Multi-objective rewards, for cooperative agents."""
        agents_rewards = [
            self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles
        ]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards)
            / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }
    

    # MODIFIED: Complex reward
    # def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
    #     """Combine reward components using weights."""
    #     rewards = self._agent_rewards(action, vehicle)
    #     weights = self.config["reward_weights"]
        
    #     # Combine rewards with weights
    #     reward = (
    #         weights["collision"] * rewards["collision_reward"] +
    #         weights["speed"] * rewards["speed_reward"] +
    #         weights["distance"] * rewards["distance_reward"] +
    #         weights["comfort"] * rewards["comfort_reward"]
    #     )
        
    #     # Override with arrival reward if arrived
    #     if rewards["arrived_reward"]:
    #         reward = weights["arrival"]
        
    #     # Zero reward if off road
    #     reward *= weights["on_road"] * rewards["on_road_reward"]
        
    #     return reward

    # ORIGINAL: Simple reward
    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        reward = 0.0
        weights = self.config["reward_weights"]

        if vehicle.crashed:
            return weights["collision"]
        if self.has_arrived(vehicle):
            return weights["arrival"]
        
        # Speed reward: quadratic scaling to incentivize max speed
        speed_ratio = vehicle.speed / self.config["max_speed"]
        reward += weights["speed"] * speed_ratio ** 2  # 2.0 * (speed/9)^2

        # Progressive proximity penalty
        min_distance = min(
            (np.linalg.norm(vehicle.position - other.position) 
            for other in self.road.vehicles if other is not vehicle),
            default=float('inf')
        )
        if min_distance < self.config["min_safe_distance"]:
            penalty = (self.config["min_safe_distance"] - min_distance) ** 2  # Squared penalty
            reward += weights["proximity_penalty"] * penalty

        if not vehicle.on_road:
            reward += weights["on_road"] * -1

        return reward

    
    # # MODIFIED: Complex reward
    # def _agent_rewards(self, action: int, vehicle: Vehicle) -> Dict[Text, float]:
    #     """Enhanced reward components for ego vehicle."""
    #     # Basic speed reward
    #     scaled_speed = utils.lmap(
    #         vehicle.speed, self.config["reward_speed_range"], [0, 1]
    #     )
        
    #     # Distance reward (safety)
    #     min_distance = float('inf')
    #     for other in self.road.vehicles:
    #         if other is not vehicle:
    #             distance = np.linalg.norm(vehicle.position - other.position)
    #             min_distance = min(min_distance, distance)
        
    #     distance_reward = -1.0 if min_distance < self.config["min_safe_distance"] else 0.0
        
    #     # Comfort reward (based on acceleration)
    #     comfort_reward = 0.0
    #     if hasattr(vehicle, 'action'):
    #         acceleration = abs(vehicle.action['acceleration'])
    #         if acceleration > self.config["comfort_accel_range"]:
    #             comfort_reward = -1.0
        
    #     # Speed matching reward
    #     speed_reward = -abs(vehicle.speed - self.config["target_speed"]) / self.config["target_speed"]
        
    #     return {
    #         "collision_reward": vehicle.crashed,
    #         "speed_reward": speed_reward,
    #         "distance_reward": distance_reward,
    #         "comfort_reward": comfort_reward,
    #         "arrived_reward": self.has_arrived(vehicle),
    #         "on_road_reward": vehicle.on_road,
    #     }

    # MODIFIED: Simple reward
    def _agent_rewards(self, action: int, vehicle: Vehicle) -> Dict[Text, float]:
        """Ego vehicle reward components."""
        scaled_speed = utils.lmap(
            vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": vehicle.crashed,  # Large negative reward for crashes
            "high_speed_reward": np.clip(scaled_speed, 0, 1),  # Reward for maintaining good speed
            "arrived_reward": self.has_arrived(vehicle),  # Reward for reaching destination
            "on_road_reward": vehicle.on_road,  # Must stay on road
        }

    # ORIGINAL: Simple reward
    # def _agent_rewards(self, action: int, vehicle: Vehicle) -> Dict[Text, float]:
    #     """Per-agent per-objective reward signal."""
    #     scaled_speed = utils.lmap(
    #         vehicle.speed, self.config["reward_speed_range"], [0, 1]
    #     )
    #     return {
    #         "collision_reward": vehicle.crashed,
    #         "high_speed_reward": np.clip(scaled_speed, 0, 1),
    #         "arrived_reward": self.has_arrived(vehicle),
    #         "on_road_reward": vehicle.on_road,
    #     }

    def _is_terminated(self) -> bool:
        return (
            any(vehicle.crashed for vehicle in self.controlled_vehicles)
            or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles)
            or (self.config["offroad_terminal"] and not self.vehicle.on_road)
        )

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed or self.has_arrived(vehicle)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        )
        info["agents_dones"] = tuple(
            self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles
        )
        return info

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, truncated, info

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            # Incoming
            start = rotation @ np.array(
                [lane_width / 2, access_length + outer_distance]
            )
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane(
                "o" + str(corner),
                "ir" + str(corner),
                StraightLane(
                    start, end, line_types=[s, c], priority=priority, speed_limit=10
                ),
            )
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner - 1) % 4),
                CircularLane(
                    r_center,
                    right_turn_radius,
                    angle + np.radians(180),
                    angle + np.radians(270),
                    line_types=[n, c],
                    priority=priority,
                    speed_limit=10,
                ),
            )
            # Left turn
            l_center = rotation @ (
                np.array(
                    [
                        -left_turn_radius + lane_width / 2,
                        left_turn_radius - lane_width / 2,
                    ]
                )
            )
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 1) % 4),
                CircularLane(
                    l_center,
                    left_turn_radius,
                    angle + np.radians(0),
                    angle + np.radians(-90),
                    clockwise=False,
                    line_types=[n, n],
                    priority=priority - 1,
                    speed_limit=10,
                ),
            )
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 2) % 4),
                StraightLane(
                    start, end, line_types=[s, n], priority=priority, speed_limit=10
                ),
            )
            # Exit
            start = rotation @ np.flip(
                [lane_width / 2, access_length + outer_distance], axis=0
            )
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane(
                "il" + str((corner - 1) % 4),
                "o" + str((corner - 1) % 4),
                StraightLane(
                    end, start, line_types=[n, c], priority=priority, speed_limit=10
                ),
            )

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road
    
    def _make_vehicles_editted(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with vehicles at specific positions:
        - 2 vehicles from west (left)
        - 2 vehicles from east (right)
        - 1 vehicle from north
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 3  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Create vehicles at specific positions
        # Format: (start_position, route_start, route_end, longitudinal_pos, speed)
        # route_start: 0=south, 1=west, 2=north, 3=east
        specific_vehicles = [
            # Two vehicles from west (1)
            (0, 1, 3, 40, 8),  # First vehicle from west going straight
            (0, 1, 3, 60, 8),  # Second vehicle from west going straight
            (0, 1, 3, 20, 8),  # Second vehicle from west going straight
            (0, 1, 3, 80, 8),  # Second vehicle from west going straight
            (0, 1, 3, 100, 8),  # Second vehicle from west going straight
            (0, 1, 3, 120, 8),  # Second vehicle from west going straight

            # Two vehicles from east (3)
            (0, 3, 1, 40, 8),  # First vehicle from east going straight
            (0, 3, 1, 60, 8),  # Second vehicle from east going straight
            
            # One vehicle from north (2)
            (0, 2, 0, 50, 8),  # Vehicle from north going straight
        ]

        for spawn_time, route_start, route_end, longitudinal_pos, speed in specific_vehicles:
            vehicle = vehicle_type.make_on_lane(
                self.road,
                ("o" + str(route_start), "ir" + str(route_start), 0),
                longitudinal=longitudinal_pos,
                speed=speed
            )
            vehicle.plan_route_to("o" + str(route_end))
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        # Simulation steps to let vehicles get into position
        simulation_steps = 3
        for _ in range(simulation_steps):
            [
                (
                    self.road.act(),
                    self.road.step(1 / self.config["simulation_frequency"]),
                )
                for _ in range(self.config["simulation_frequency"])
            ]

        # Controlled vehicles (ego vehicle)
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(
                ("o{}".format(ego_id % 4), "ir{}".format(ego_id % 4), 0)
            )
            destination = self.config["destination"] or "o" + str(
                self.np_random.integers(1, 4)
            )
            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                ego_lane.position(60, 0),  # Fixed position for ego vehicle
                speed=ego_lane.speed_limit,
                heading=ego_lane.heading_at(60),
            )
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [
                (
                    self.road.act(),
                    self.road.step(1 / self.config["simulation_frequency"]),
                )
                for _ in range(self.config["simulation_frequency"])
            ]

        # Challenger vehicle
        self._spawn_vehicle(
            60,
            spawn_probability=1,
            go_straight=True,
            position_deviation=0.1,
            speed_deviation=0,
        )

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(
                ("o{}".format(ego_id % 4), "ir{}".format(ego_id % 4), 0)
            )
            destination = self.config["destination"] or "o" + str(
                self.np_random.integers(1, 4)
            )
            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                ego_lane.position(60 + 5 * self.np_random.normal(1), 0),
                speed=ego_lane.speed_limit,
                heading=ego_lane.heading_at(60),
            )
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(
                    ego_lane.speed_limit
                )
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(
                    ego_vehicle.speed_index
                )
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if (
                    v is not ego_vehicle
                    and np.linalg.norm(v.position - ego_vehicle.position) < 20
                ):
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(
        self,
        longitudinal: float = 0,
        position_deviation: float = 1.0,
        speed_deviation: float = 1.0,
        spawn_probability: float = 0.6,
        go_straight: bool = False,
    ) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(
            self.road,
            ("o" + str(route[0]), "ir" + str(route[0]), 0),
            longitudinal=(
                longitudinal + 5 + self.np_random.normal() * position_deviation
            ),
            speed=8 + self.np_random.normal() * speed_deviation,
        )
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = (
            lambda vehicle: "il" in vehicle.lane_index[0]
            and "o" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0]
            >= vehicle.lane.length - 4 * vehicle.LENGTH
        )
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle in self.controlled_vehicles
            or not (is_leaving(vehicle) or vehicle.route is None)
        ]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return (
            "il" in vehicle.lane_index[0]
            and "o" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
        )


class MultiAgentIntersectionEnv(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
                        "lateral": False,
                        "longitudinal": True,
                    },
                },
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {"type": "Kinematics"},
                },
                "controlled_vehicles": 2,
            }
        )
        return config


class ContinuousIntersectionEnv(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 5,
                    "features": [
                        "presence",
                        "x",
                        "y",
                        "vx",
                        "vy",
                        "long_off",
                        "lat_off",
                        "ang_off",
                    ],
                },
                "action": {
                    "type": "ContinuousAction",
                    "steering_range": [-np.pi / 3, np.pi / 3],
                    "longitudinal": True,
                    "lateral": True,
                    "dynamical": True,
                },
            }
        )
        return config
