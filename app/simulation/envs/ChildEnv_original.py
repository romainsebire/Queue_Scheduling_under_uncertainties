from app.simulation.envs.Env import Env
from app.domain.Customer import Customer

class ChildEnv(Env):
    def _get_action_space(self):
        """
        Get the action space.

        Returns:
            action space compatible with gymnasium
        """
        raise NotImplementedError
    
    def _get_observation_space(self):
        """
        Get the observation space.

        Returns: 
            observation space compatible with gymnasium
        """
        raise NotImplementedError
    
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
        raise NotImplementedError
    
    def _get_customer_from_action(self, action) -> Customer:
        """
        Return customer from action.

        Retunrs:
            Customer, or None if invalid action. 
        """   
        # Customers mus be taken from the customers waiting
        # Waiting Customers: dict{customer_id: customer}
        #   customer attributes: id: int, arrival_time: float, task_id: int
        self.customer_waiting
        raise NotImplementedError    

    def _get_invalid_action_reward(self) -> float: 
        """
        Reward chosen for invalid action.

        Returns:
            reward (float) 
        """  
        # ex: return -10
        raise NotImplementedError 
    
    def _get_valid_reward(self, customer: Customer) -> float:
        """
        Get valid reward.

        Parameters:
            customer (Customer): customer chosen by the action.

        Returns:
            reward (float)
        """
        # ex: return 10
        raise NotImplementedError
    
    def action_masks(self):
        """
        Mask not accepted actions.
        """
        raise NotImplementedError
    
    def _get_hold_action_number(self):
        """
        Get the action to tell the server to hold and not assign a customer.
        """
        raise NotImplementedError