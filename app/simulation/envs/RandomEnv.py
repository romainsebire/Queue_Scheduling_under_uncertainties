from app.simulation.envs.Env import Env
from app.domain.Customer import Customer
import gymnasium as gym
import random
import numpy as np

random_size = 10

class RandomEnv(Env):
    def _get_action_space(self):
        """
        Get the action space.

        Returns:
            action space compatible with gymnasium
        """
        return gym.spaces.Discrete(random_size)
    
    def _get_observation_space(self):
        """
        Get the observation space.

        Returns: 
            observation space compatible with gymnasium
        """
        return gym.spaces.Box(
                    low=0,       
                    high=5000,
                    shape=(random_size,),
                    dtype=int
                )
    
    def _get_obs(self):
        """
        Convert internal state to observation format.

        Returns: 
            obs(np.array)
        """
        ### Data that can be extracted from the space:
        # Waiting Customers: dict{customer_id: customer}
        #   customer attributes: id: int, arrival_time: float, task_id: int
        # Appointments (all, even passed): dict{customer_id: appointement}
        #   appointment attributes: time: float, customer_id: int, task_id: int, service_time: float
        # Servers: dict{server_id: server}
        #   server attributes: id: int, avg_service_time: dict{task_id: float}
        # Expected end of server activity: dict{server_id: float}, if 0, server available
        # Current selected server id (int)
        # Current simulation time (float)
        waiting_customers, appointments, servers, expected_end, selected_server_id, sim_time = self._get_state()
        customers_id = list(waiting_customers.keys())

        if len(customers_id) >= random_size:
            sampled = random.sample(customers_id, random_size)
        else:
            sampled = customers_id + [0] * (random_size - len(customers_id))
        return np.array(sampled)
    
    def _get_customer_from_action(self, action) -> Customer:
        """
        Return customer from action.

        Retunrs:
            Customer, or None if invalid action. 
        """   
        # Customers mus be taken from the customers waiting
        # Waiting Customers: dict{customer_id: customer}
        #   customer attributes: id: int, arrival_time: float, task_id: int
        return self.customer_waiting[action]
         

    def _get_invalid_action_reward(self) -> float: 
        """
        Reward chosen for invalid action.

        Returns:
            reward (float) 
        """  
        # ex: return -10
        return -10
    
    def _get_valid_reward(self, customer: Customer) -> float:
        """
        Get valid reward.

        Parameters:
            customer (Customer): customer chosen by the action.

        Returns:
            reward (float)
        """
        # ex: return 10
        return 10
    
    def action_masks(self):
        """
        Mask not accepted actions.
        """
        customers_id = list(self.customer_waiting.keys())

        action_mask = (
            [True] * random_size
            if len(customers_id) >= random_size
            else [True] * len(customers_id) + [False] * (random_size - len(customers_id))
        )
        return action_mask

    
    def _get_hold_action_number(self):
        """
        Get the action to tell the server to hold and not assign a customer.
        """
        return -1