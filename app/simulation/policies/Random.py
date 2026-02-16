from app.simulation.policies.Policy import Policy
from app.simulation.envs.Env import Env
import gymnasium as gym
import random

class Random(Policy):
    def _predict(self, obs, info):
        action_mask = info.get('action_mask')
        valid_indices = [i for i, m in enumerate(action_mask) if m]
        idx = random.choice(valid_indices)
        random_id = obs[idx]
        
        return random_id
    
    def learn(self, scenario, total_timesteps, verbose):
        """
        Learning and intialization method for learning models.
        For other models, does nothing.

        Parameters: 
            scenario (scenario): scenario to train
            total_timesteps(int): number of time steps to learn
            verbose(bool): show logs 
        """
        #learning_env = gym.make("Random_Env", mode=Env.MODE.TRAIN, scenario=scenario)
        pass