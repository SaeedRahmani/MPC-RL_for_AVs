import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 64):
        # Handle both Dict and non-Dict observation spaces
        if isinstance(observation_space, spaces.Dict):
            self.structured_space = observation_space.spaces['structured']
        else:
            self.structured_space = observation_space
            
        super().__init__(observation_space, features_dim)
        
        # Get input dimension from the structured space
        n_input_features = get_flattened_obs_dim(self.structured_space)
        
        self.feature_net = nn.Sequential(
            nn.Linear(n_input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
        )

    def forward(self, observations):
        # Extract structured observations if dictionary
        if isinstance(observations, dict):
            features = observations['structured']
        else:
            features = observations
            
        # Ensure observations is a float tensor
        if not isinstance(features, th.Tensor):
            features = th.as_tensor(features).float()
            
        return self.feature_net(features.reshape(features.shape[0], -1))