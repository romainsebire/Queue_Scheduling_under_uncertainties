import numpy as np
import gymnasium as gym
from gymnasium import spaces
from app.simulation.envs.Env import Env
from app.domain.Customer import Customer

class ChildEnv(Env):
    def __init__(self, mode, instance=None, scenario=None):
        # Sizing parameters for RL
        self.max_obs_queue = 50  # The agent only "sees" and can choose among the first 50 customers
        self.num_server_features = 2  # [Busy?, Time_until_free] + avg_service_times
        self.num_queue_features = 6   # [WaitTime, TaskID, IsAppt?, ApptDelta, EstServiceTime, AbandonTime]
        
        # List to map the action index (0, 1...) to the real customer ID
        self.queue_mapping = [] 
        
        super().__init__(mode, instance, scenario)

    def _get_action_space(self):
        """
        Action space:
        0 to max_obs_queue-1 : Choose the customer at this index in the sorted queue.
        max_obs_queue        : HOLD action (do nothing).
        """
        # +1 for the HOLD action
        return spaces.Discrete(self.max_obs_queue + 1)
    
    def _get_observation_space(self):
        """
        Observation space composed of:
        - queue   : Matrix (N_max, Features)
        - servers : Matrix (N_servers, Features)
        - context : Vector (Current_Time, Current_Server_ID)
        """
        # Queue observation space
        queue_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.max_obs_queue, self.num_queue_features), 
            dtype=np.float32
        )
        
        # Server observation space
        # Features: [Is_Busy, Remaining_Time, Avg_Time_Task_0, Avg_Time_Task_1, ...]
        server_feat_count = 2 + self.num_needs
        server_space = spaces.Box(
            low=0, high=np.inf, 
            shape=(self.c, server_feat_count), 
            dtype=np.float32
        )
        
        # Global context (Time, ID of the server that must choose)
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
        """
        Vectorizes the simulation state.
        This is where self.queue_mapping is updated for the next step.
        """
        waiting_customers, appointments, servers, expected_end, selected_server_id, sim_time = self._get_state()
        
        # --- 1. Queue processing ---
        # Sort customers by arrival time (FIFO) to stabilize the network input
        sorted_customers = sorted(waiting_customers.values(), key=lambda c: c.arrival_time)
        
        # Update mapping for the next action
        self.queue_mapping = [c.id for c in sorted_customers[:self.max_obs_queue]]
        
        queue_matrix = np.zeros((self.max_obs_queue, self.num_queue_features), dtype=np.float32)
        
        for i, customer in enumerate(sorted_customers):
            if i >= self.max_obs_queue:
                break
            
            # Feature extraction
            wait_time = sim_time - customer.arrival_time
            is_appt = 1.0 if customer.id in appointments else 0.0
            
            appt_delta = 0.0
            if is_appt:
                appt_delta = sim_time - appointments[customer.id].time
            
            # Basic estimation of service time (average over all servers for this task)
            # Could be refined using the average time of the selected server
            task_avg_times = [s.avg_service_time[customer.task] for s in servers.values()]
            est_service = sum(task_avg_times) / len(task_avg_times) if task_avg_times else 0
            
            time_before_abandon = customer.abandonment_time - sim_time

            queue_matrix[i] = [
                wait_time, 
                float(customer.task), 
                is_appt, 
                appt_delta, 
                est_service,
                time_before_abandon
            ]

        # --- 2. Server processing ---
        # Features: [Is_Busy (0/1), Time_Until_Free, Capacity_Task_0, Capacity_Task_1...]
        server_matrix = np.zeros((self.c, 2 + self.num_needs), dtype=np.float32)
        
        for s_id in range(self.c):
            # Is this server busy?
            is_busy = 1.0 if expected_end[s_id] > sim_time else 0.0
            remaining_time = max(0.0, expected_end[s_id] - sim_time)
            
            server_matrix[s_id, 0] = is_busy
            server_matrix[s_id, 1] = remaining_time
            
            # Capacities (average service times)
            server_obj = servers[s_id]
            for task_id in range(self.num_needs):
                server_matrix[s_id, 2 + task_id] = server_obj.avg_service_time.get(task_id, 0.0)

        # --- 3. Context ---
        context = np.array([sim_time, float(selected_server_id)], dtype=np.float32)

        return {
            "queue": queue_matrix,
            "servers": server_matrix,
            "context": context
        }
    
    def _get_customer_from_action(self, action) -> Customer:
        """
        Retrieves the Customer object using the mapping created during the previous observation.
        """
        if action == self._get_hold_action_number():
            return None
        
        if action >= len(self.queue_mapping):
            return None  # Invalid action (empty index)
            
        customer_id = self.queue_mapping[action]
        return self.customer_waiting.get(customer_id)

    def _get_hold_action_number(self):
        return self.max_obs_queue

    def action_masks(self):
        """
        Returns a boolean mask [True, True, False, ..., True]
        indicating which actions are valid.
        """
        mask = [False] * (self.max_obs_queue + 1)
        
        # Indices corresponding to real customers in the mapping are valid
        num_customers = len(self.queue_mapping)
        for i in range(num_customers):
            # A condition could be added here: can the selected server handle this task?
            # For now, we assume yes, or that it's handled by _check_existing_possible_service in Env
            mask[i] = True
            
        # HOLD action is always valid (unless forcing an action)
        mask[self._get_hold_action_number()] = True
        
        return mask

    def _get_invalid_action_reward(self) -> float: 
        """
        The agent tried to serve a ghost (empty index).
        """
        return -10.0
    
    def _get_valid_reward(self, customer: Customer) -> float:
        """
        Reward computed immediately after assignment.
        We try to approximate the final metric.
        """
        reward = 0.0
        
        # 1. Base reward for serving someone (vs HOLD)
        reward += 10.0
        
        # 2. Penalty on waiting time
        # We want to minimize wi/Wmax
        # If unbearable_wait is defined in the scenario, use it
        w_max = self.scenario.unbearable_wait if hasattr(self, 'scenario') else 60.0
        if w_max == 0: w_max = 60.0
        
        wait_time = self.system_time - customer.arrival_time
        
        # Formula inspired by evaluation: 100*(1 - wi/Wmax)
        # We give this "score" as immediate reward
        wait_score = 100 * (1 - (wait_time / w_max))
        reward += wait_score * 0.4  # Weight 0.4 as in evaluation
        
        # 3. Appointment reward
        if customer.id in self.appointments:
            appt = self.appointments[customer.id]
            delta = self.system_time - appt.time  # Positive = late
            
            # If exactly on time (delta near 0) → big bonus
            # If very late or very early → penalty
            epsilon = 3.0  # Tolerance
            
            if abs(delta) <= epsilon:
                reward += 100 * 0.4  # Max weighted score
            elif delta > epsilon:
                # Late
                # Simple linear penalty for RL training
                reward -= min(50, delta) * 0.5 
            else:
                # Too early (delta < -3)
                reward -= min(50, abs(delta)) * 0.5

        return reward
