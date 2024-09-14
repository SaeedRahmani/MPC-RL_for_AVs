import gymnasium as gym
from ..utils.typing import Action, RawObservation, ParsedObservation

class Agent:
    """
    Base class for agents to train or simulate in the
    `intersection` scenario.
    """
    
    def __init__(
        self, 
        env: gym.Env
    ):
        self.env = env.unwrapped
        self.observation_dim = env.observation_space.shape
        self.max_observed_agents: int = self.observation_dim[0]
        self.num_observed_properties: int = self.observation_dim[1]
    
    def __str__(self) -> str:
        return f"Agent"
    
    def __repr__(self) -> str:
        return f"Agent"
    
    def make_decision(observation: RawObservation) -> Action:
        raise NotImplementedError()
    
    def parse_observation(observation: RawObservation) -> ParsedObservation:
        raise NotImplemented