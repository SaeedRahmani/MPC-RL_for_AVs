import torch as th
from torch import nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim


class VehicleAttentionNetwork(nn.Module):
    def __init__(self, n_vehicles, feature_dim):
        super().__init__()
        
        # Dimensions
        self.n_vehicles = n_vehicles
        self.feature_dim = feature_dim
        self.hidden_dim = 64
        
        # Embedding layers
        self.ego_embedding = nn.Linear(8, self.hidden_dim)  # Ego vehicle
        self.vehicle_embedding = nn.Linear(12, self.hidden_dim)  # Other vehicles (relative pos + vel + features)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Post-attention processing
        self.post_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Split features
        ego_features = x[:, :8]  # First 8 features are ego vehicle
        other_features = x[:, 8:].view(batch_size, self.n_vehicles-1, -1)
        
        # Embed features
        ego_embedded = self.ego_embedding(ego_features).unsqueeze(1)  # [batch, 1, hidden_dim]
        others_embedded = self.vehicle_embedding(other_features)  # [batch, n_vehicles-1, hidden_dim]
        
        # Combine embeddings
        all_embedded = torch.cat([ego_embedded, others_embedded], dim=1)  # [batch, n_vehicles, hidden_dim]
        
        # Apply attention
        attended, _ = self.attention(all_embedded, all_embedded, all_embedded)
        
        # Process attention output
        output = self.post_attention(attended.mean(dim=1))  # Mean pooling over vehicles
        
        return output
    
class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor Critic policy for handling structured observations.
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # Network architecture
        net_arch = {
            'pi': [128, 64],  # Policy network
            'vf': [128, 64]   # Value function network
        }
        
        # Add features extractor kwargs
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = CustomFeaturesExtractor
        if "features_extractor_kwargs" not in kwargs:
            kwargs["features_extractor_kwargs"] = dict(features_dim=64)
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            **kwargs
        )

    def extract_features(self, obs):
        """
        Extract features from the observation.
        
        :param obs: Dictionary or tensor observation
        :return: Processed features
        """
        if isinstance(obs, dict):
            return obs['structured']
        return obs

    def forward(self, obs, deterministic: bool = False):
        """
        Forward pass in all the networks.
        
        :param obs: Observation dictionary or tensor
        :param deterministic: Whether to sample or use deterministic actions
        :return: Actions, values, and log probabilities
        """
        features = self.extract_features(obs)
            
        # Get actions from policy network
        mean_actions = self.policy_net(features)
        
        # Handle continuous vs discrete actions
        if isinstance(self.action_space, gym.spaces.Box):
            if deterministic:
                actions = mean_actions
            else:
                noise = th.randn_like(mean_actions) * self.log_std.exp()
                actions = mean_actions + noise
                
            log_prob = -0.5 * (
                ((actions - mean_actions) ** 2 / (self.log_std.exp() ** 2))
                + 2 * self.log_std
                + np.log(2 * np.pi)
            ).sum(dim=1)
        else:
            # For discrete actions
            actions = mean_actions
            dist = self.get_distribution(features)
            log_prob = dist.log_prob(actions)
            
        # Get values from value network
        values = self.value_net(features)
        
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the current policy given the observations.
        
        :param obs: Observation dictionary or tensor
        :param actions: Actions to evaluate
        :return: Values, log probability, and entropy of the action distribution
        """
        features = self.extract_features(obs)
            
        values = self.value_net(features)
        mean_actions = self.policy_net(features)
        
        if isinstance(self.action_space, gym.spaces.Box):
            log_prob = -0.5 * (
                ((actions - mean_actions) ** 2 / (self.log_std.exp() ** 2))
                + 2 * self.log_std
                + np.log(2 * np.pi)
            ).sum(dim=1)
            
            # Compute entropy
            entropy = 0.5 + 0.5 * np.log(2 * np.pi) + self.log_std
            entropy = entropy.sum(dim=-1)
        else:
            dist = self.get_distribution(features)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
            
        return values, log_prob, entropy

    def predict_values(self, obs):
        """
        Get the estimated values according to the current policy given the observations.
        
        :param obs: Observation dictionary or tensor
        :return: The estimated values
        """
        features = self.extract_features(obs)
        return self.value_net(features)