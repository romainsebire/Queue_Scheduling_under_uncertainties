from abc import ABC, abstractmethod
from app.utils import plot_gantt, save_client_history_to_csv
import time

class Policy(ABC):
    def __init__(self, model_title):
        """
        Initialization.
        
        Parameters:
            model_title(str): Title of the model
        """
        super().__init__()
        self.model_title = model_title

    @abstractmethod
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
        pass
    
    def simulate(self, env, print_logs: bool = False, 
                 save_to_csv: bool = False, path: str = None, file_name: str = None):
        """
        Simulate the given environment and call the action of the model

        Parameters:
            env (Env): Simulation environment
            print_logs (bool): print the observations at each step
                                Default: False
        """

        if save_to_csv:
            assert path is not None, "To print to csv, a path should be given."
            assert file_name is not None, "To print to csv, a file_name should be given."

        self.env = env
        self.print_logs = print_logs

        self.obs, self.info = self.env.reset(seed=42)
        done = False
        self.total_reward = 0
        start_time = time.time()
        while not done:
            action = self._predict(self.obs, self.info)
            next_observation, reward, self.terminated, self.truncated, self.info = self.env.step(action)
            if print_logs:
                print("Timestep ", self.info.get("system_time"))
                print("Observation ", self.obs)
                print("Sampled action ", action)
                print("Reward ", reward)
                print("Queue length ", self.info.get("queue_length"))
            self.obs = next_observation
            self.total_reward += reward
            done = self.terminated or self.truncated
            if done: 
                end_time = time.time()
                self.execution_time = end_time - start_time
                self.total_number_of_customers = self.info.get("total_number_of_customers")
                self.customer_abandonment = self.info.get("customer_abandonment")
                self.avg_waiting_time = self.info.get("avg_waiting_time")
                self.unserved_customers = self.total_number_of_customers - self.info.get("served_clients")
                self.customer_on_service = self.info.get("customers_on_service")
                self.customers_history = self.info.get("served_clients_info")
                self.env.reset()
        if print_logs:
            print(f"\n--- FINAL RESULTS ---")
            print(f"Total number of arrived customers = {self.total_number_of_customers}")
            print(f"Simulation truncated: {'Yes 'if self.truncated else 'No'}")
            print(f"Total reward = {self.total_reward}")
            print(f"Number of abandonment = {self.customer_abandonment}")
            print(f"Average waiting time = {self.avg_waiting_time}")
            print(f"Number of unserved clients = {self.unserved_customers}")
            print(f"Servers on service at the end = {self.customer_on_service}")
            plot_gantt(self.customers_history, self.env.unwrapped.c, title=self.model_title)
        if save_to_csv: 
            save_client_history_to_csv(self.customers_history, path, file_name)
            


