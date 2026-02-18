import numpy as np
import gymnasium as gym
from gymnasium import spaces
from app.simulation.envs.Env import Env
from app.domain.Customer import Customer

class ChildEnv(Env):
    def __init__(self, mode, instance=None, scenario=None):
        self.max_obs_queue = 50 
        self.num_server_features = 2  # [busy, time_until_free] + avg_service_times
        self.num_queue_features = 6   # [wait_time, task_id, is_appt, appt_delta, est_service_time, abandon_time]
        
        # map action index to customer id
        self.queue_mapping = [] 
        
        super().__init__(mode, instance, scenario)

    def _get_action_space(self):
        """
        0 to max_obs_queue-1 : choose customer at index
        max_obs_queue        : hold action
        """
        return spaces.Discrete(self.max_obs_queue + 1)
    
    def _get_observation_space(self):
        # queue matrix
        queue_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.max_obs_queue, self.num_queue_features), 
            dtype=np.float32
        )
        
        # server matrix
        # features: [is_busy, remaining_time, avg_time_task_0, avg_time_task_1, ...]
        server_feat_count = 2 + self.num_needs
        server_space = spaces.Box(
            low=0, high=np.inf, 
            shape=(self.c, server_feat_count), 
            dtype=np.float32
        )
        
        # context vector: [time, current_server_id]
        context_space = spaces.Box(
            low=0, high=np.inf, 
            shape=(2,), 
            dtype=np.float32
        )

        return spaces.Dict({
            "queue": queue_space,
            "servers": server_space,
            "context": context_space
        })
    
    def _get_obs(self):
        waiting_customers, appointments, servers, expected_end, selected_server_id, sim_time = self._get_state()
        
        # heuristic sorting strategy
        def priority_key(c):
            # group 1: patients with appointment
            if c.id in appointments:
                return -1000000000 + appointments[c.id].time
            
            # group 2: walk-in (fifo)
            return c.arrival_time

        sorted_customers = sorted(waiting_customers.values(), key=priority_key)
        
        # update mapping
        self.queue_mapping = [c.id for c in sorted_customers[:self.max_obs_queue]]
        
        queue_matrix = np.zeros((self.max_obs_queue, self.num_queue_features), dtype=np.float32)
        
        w_max = self.scenario.unbearable_wait if hasattr(self, 'scenario') else 60.0
        if w_max == 0: w_max = 60.0

        for i, customer in enumerate(sorted_customers):
            if i >= self.max_obs_queue: break
            
            wait_time = (sim_time - customer.arrival_time) / w_max
            
            appt_delta = 0.0
            is_appt = 0.0
            if customer.id in appointments:
                is_appt = 1.0
                raw_delta = sim_time - appointments[customer.id].time
                appt_delta = np.clip(raw_delta / 60.0, -1.0, 1.0) 
            
            # estimate service duration
            avg_duration = 10.0 
            potential_times = [s.avg_service_time.get(customer.task, 0) for s in servers.values()]
            valid_times = [t for t in potential_times if t > 0]
            if valid_times:
                avg_duration = sum(valid_times) / len(valid_times)
            est_service = np.clip(avg_duration / 30.0, 0.0, 2.0)

            time_before_abandon = 1.0 
            if customer.abandonment_time is not None:
                raw_left = customer.abandonment_time - sim_time
                time_before_abandon = 1.0 - np.clip(raw_left / w_max, 0.0, 1.0)

            queue_matrix[i] = [
                wait_time, 
                float(customer.task) / self.num_needs, 
                is_appt, 
                appt_delta, 
                est_service,
                time_before_abandon
            ]

        server_matrix = np.zeros((self.c, 2 + self.num_needs), dtype=np.float32)
        for s_id in range(self.c):
            is_busy = 1.0 if expected_end[s_id] > sim_time else 0.0
            remaining = max(0.0, expected_end[s_id] - sim_time) / 30.0 
            
            server_matrix[s_id, 0] = is_busy
            server_matrix[s_id, 1] = remaining
            
            for task_id in range(self.num_needs):
                val = servers[s_id].avg_service_time.get(task_id, 0.0)
                server_matrix[s_id, 2 + task_id] = val / 30.0

        context = np.array([sim_time / self.max_sim_time, float(selected_server_id) / self.c], dtype=np.float32)

        return {
            "queue": queue_matrix,
            "servers": server_matrix,
            "context": context
        }
    
    def _get_customer_from_action(self, action) -> Customer:
        if action == self._get_hold_action_number():
            return None
        
        if action >= len(self.queue_mapping):
            return None 
            
        customer_id = self.queue_mapping[action]
        return self.customer_waiting.get(customer_id)

    def _get_hold_action_number(self):
        return self.max_obs_queue

    def action_masks(self):
        mask = [False] * (self.max_obs_queue + 1)
        current_server = self.current_working_server
        
        if current_server is None: return mask 

        num_customers = len(self.queue_mapping)
        can_serve_someone = False
        
        for i in range(num_customers):
            cust_id = self.queue_mapping[i]
            customer = self.customer_waiting.get(cust_id)
            
            if customer:
                # check competence
                if current_server.avg_service_time.get(customer.task, 0) > 0:
                    mask[i] = True
                    can_serve_someone = True
        
        # forbid hold if service is possible
        if can_serve_someone:
            mask[self._get_hold_action_number()] = False
        else:
            mask[self._get_hold_action_number()] = True
        
        return mask

    def _get_invalid_action_reward(self) -> float: 
        return -10.0
    
    def _get_valid_reward(self, customer: Customer) -> float:
        reward = 0.0
        
        # base throughput
        reward += 15.0
        
        # appointment bonus
        if customer.id in self.appointments:
            appt = self.appointments[customer.id]
            delta = abs(self.system_time - appt.time)
            
            if delta <= 5.0:       
                reward += 40.0  
            elif delta <= 30.0:    
                reward += 15.0  
            else:                  
                reward += 0.0   
        
        # walk-in bonus
        else:
            w_max = self.scenario.unbearable_wait if hasattr(self, 'scenario') else 60.0
            wait_ratio = (self.system_time - customer.arrival_time) / w_max
            
            if wait_ratio < 1.0:
                reward += 5.0 * (1.0 - wait_ratio)

        return reward