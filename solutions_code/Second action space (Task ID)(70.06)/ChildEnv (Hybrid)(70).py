from app.simulation.envs.Env import Env
from app.domain.Customer import Customer
from gymnasium import spaces
import numpy as np

class ChildEnv(Env):
    def __init__(self, mode, instance=None, scenario=None):
        if scenario:
            self.limit_wait = scenario.unbearable_wait
            self.limit_vip_delay = getattr(scenario, 'unbearable_wait_appointement', 30.0)
        else:
            self.limit_wait = 60.0
            self.limit_vip_delay = 30.0
            
        self.limit_vip_early = 60.0

        super().__init__(mode, instance, scenario)

    def _get_action_space(self):
        # actions: 0 to num_needs-1 (tasks), num_needs (patient with an appointment), num_needs+1 (wait)
        return spaces.Discrete(self.num_needs + 2)

    def _get_observation_space(self):
        # size = (num_needs * 3) + 4
        # [count, max_wait, efficiency] per task
        # + [appt_count, appt_urgency, time, my_id]
        obs_size = (self.num_needs * 3) + 4
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def _get_hold_action_number(self):
        # wait action is the last one
        return self.num_needs + 1

    def _get_invalid_action_reward(self):
        return -10.0

    def _get_customer_from_action(self, action):
        """
        the agent provides a category (action), we find the patient.
        """
        vip_action = self.num_needs
        wait_action = self.num_needs + 1
        
        # exact parameters
        unbearable_wait = self.limit_wait 
        unbearable_appt = self.limit_vip_delay 
        max_early = 60.0
        
        # wait
        if action == wait_action:
            return None

        # patient with an appointment
        if action == vip_action:
            candidates = []
            for c in self.customer_waiting.values():
                if c.id in self.appointments:
                    # check server competence
                    if self.current_working_server.avg_service_time.get(c.task, 0) > 0:
                        
                        appt_time = self.appointments[c.id].time
                        current_delay = self.system_time - appt_time

                        # not too early
                        if current_delay < -max_early:
                            continue 
                            
                        # not too late
                        if current_delay > unbearable_appt:
                            continue 

                        candidates.append(c)
            
            if not candidates: return None
            
            # sort: most urgent appointment (earliest time)
            return min(candidates, key=lambda c: self.appointments[c.id].time)

        # specific task (walk-in)
        task_id = action
        candidates = []
        for c in self.customer_waiting.values():
            if c.task == task_id and c.id not in self.appointments:
                
                current_wait = self.system_time - c.arrival_time
                
                # check if within waiting limit
                if current_wait < unbearable_wait:
                    candidates.append(c)
        
        if not candidates: return None
        # sort: fifo strict
        return min(candidates, key=lambda c: c.arrival_time)

    def _get_obs(self):
        waiting_customers, appointments, servers, expected_end, selected_server_id, sim_time = self._get_state()
        server = servers[selected_server_id]
        
        w_max = self.scenario.unbearable_wait if hasattr(self, 'scenario') else 60.0
        max_q = 50.0 
        
        # init vectors
        tasks_count = np.zeros(self.num_needs, dtype=np.float32)
        tasks_max_wait = np.zeros(self.num_needs, dtype=np.float32)
        my_efficiency = np.zeros(self.num_needs, dtype=np.float32)
        
        # walk-ins stats
        for c in waiting_customers.values():
            if c.id not in appointments:
                t_id = c.task
                tasks_count[t_id] += 1
                wait_t = sim_time - c.arrival_time
                if wait_t > tasks_max_wait[t_id]:
                    tasks_max_wait[t_id] = wait_t

        # server efficiency
        for t_id in range(self.num_needs):
            dur = server.avg_service_time.get(t_id, 0.0)
            if dur > 0:
                my_efficiency[t_id] = dur / 60.0
            else:
                my_efficiency[t_id] = 2.0 # incompetent
        
        # appointment stats
        vip_count = 0
        vip_urgency = 0.0
        for c in waiting_customers.values():
            if c.id in appointments:
                vip_count += 1
                delta = sim_time - appointments[c.id].time
                if delta > vip_urgency:
                    vip_urgency = delta
        
        # normalization
        tasks_count = np.clip(tasks_count / max_q, 0, 1)
        tasks_max_wait = np.clip(tasks_max_wait / w_max, 0, 2)
        norm_vip_count = np.clip(vip_count / max_q, 0, 1)
        norm_vip_urgency = np.clip(vip_urgency / 60.0, -1, 2)
        norm_time = sim_time / self.max_sim_time
        norm_id = float(selected_server_id) / self.c
        
        # assemble
        obs = []
        for t in range(self.num_needs):
            obs.extend([tasks_count[t], tasks_max_wait[t], my_efficiency[t]])
        obs.extend([norm_vip_count, norm_vip_urgency, norm_time, norm_id])
        
        return np.array(obs, dtype=np.float32)

    def action_masks(self):
        mask = [False] * (self.num_needs + 2)
        server = self.current_working_server
        if server is None: return mask
        
        waiting_customers = self.customer_waiting
        vip_action = self.num_needs
        wait_action = self.num_needs + 1
        
        limit_wait = self.limit_wait 
        limit_vip_late = self.limit_vip_delay 
        limit_vip_early = 60.0 
        
        # walk-in tasks
        for t_id in range(self.num_needs):
            if server.avg_service_time.get(t_id, 0) > 0:
                # is there a candidate?
                for c in waiting_customers.values():
                    if c.task == t_id and c.id not in self.appointments:
                        current_wait = self.system_time - c.arrival_time
                        if current_wait < limit_wait:
                            mask[t_id] = True
                            break

        # patient with an appointment
        has_vip = False
        for c in waiting_customers.values():
            if c.id in self.appointments:
                if server.avg_service_time.get(c.task, 0) > 0:
                    appt_time = self.appointments[c.id].time
                    current_delay = self.system_time - appt_time
                    
                    if -limit_vip_early <= current_delay <= limit_vip_late:
                        has_vip = True
                        break
                        
        if has_vip:
            mask[vip_action] = True
            
        # wait
        # anti-strike logic: only wait if no other action is possible
        if any(mask[:-1]): 
            mask[wait_action] = False
        else:
            mask[wait_action] = True
            
        return mask

    def _get_valid_reward(self, customer: Customer) -> float:
        # strategic reward
        reward = 20.0 # base (throughput)
        
        # patient with an appointment
        if customer.id in self.appointments:
            appt = self.appointments[customer.id]
            delta = abs(self.system_time - appt.time)
            if delta <= 15.0: reward += 50.0
            else: reward += 10.0
        
        # speed incentive
        duration = self.current_working_server.avg_service_time.get(customer.task, 100.0)
        if duration > 0:
            reward += (300.0 / duration)
            
        return reward