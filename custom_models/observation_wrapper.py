import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import numpy as np

class StructuredObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that provides both structured and original observations.
    """
    
    def __init__(self, env):
        super().__init__(env)
        n_vehicles = env.observation_space.shape[0]
        
        # Define dimensionality of structured observations
        structured_dim = (
            n_vehicles * env.observation_space.shape[1] +  # Original flattened
            (n_vehicles - 1) * 4 +  # Relative positions and velocities
            (n_vehicles - 1)        # Distances to ego vehicle
        )
        
        # Create dictionary observation space
        self.observation_space = gym.spaces.Dict({
            'original': env.observation_space,
            'structured': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(structured_dim,),
                dtype=np.float32
            )
        })

    def observation(self, obs):
        """
        Process the observation to include both original and structured formats.
        
        :param obs: Original observation from environment
        :return: Dictionary containing both observation formats
        """
        # Keep original observation
        original_obs = obs.copy()
        
        # Extract vehicle information
        ego_vehicle = obs[0]
        other_vehicles = obs[1:]
        
        # Calculate relative features
        relative_positions = other_vehicles[:, 1:3] - ego_vehicle[1:3]
        relative_velocities = other_vehicles[:, 3:5] - ego_vehicle[3:5]
        
        # Calculate distances to ego vehicle
        distances = np.linalg.norm(relative_positions, axis=1)
        
        # Sort vehicles by distance
        sort_indices = np.argsort(distances)
        relative_positions = relative_positions[sort_indices]
        relative_velocities = relative_velocities[sort_indices]
        distances = distances[sort_indices]
        
        # Create structured observation
        structured_obs = np.concatenate([
            obs.reshape(-1),           # Flattened original observation
            relative_positions.flatten(),
            relative_velocities.flatten(),
            distances
        ]).astype(np.float32)
        
        return {
            'original': th.from_numpy(original_obs).float(),
            'structured': th.from_numpy(structured_obs).float()
        }