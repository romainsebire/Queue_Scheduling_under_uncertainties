from app.simulation.policies.Policy import Policy
from app.simulation.envs.Env import Env
import gymnasium as gym

class ChildPolicy(Policy):
    def _predict(self, obs, info):
        raise NotImplementedError
    
    def learn(self, scenario, total_timesteps, verbose):
        """
        Learning and intialization method for learning models.
        For other models, does nothing.

        Parameters: 
            scenario (scenario): scenario to train
            total_timesteps(int): number of time steps to learn
            verbose(bool): show² logs 
        """
        learning_env = gym.make("Child_Env", mode=Env.MODE.TRAIN, scenario=scenario)
        raise NotImplementedError