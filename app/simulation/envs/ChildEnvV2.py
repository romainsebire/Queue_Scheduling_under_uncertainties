# Solution V2 — Action Space: Task ID + VIP + Hold (score ~70%)
# Action = task category (0..num_needs-1), VIP appointment (num_needs), hold (num_needs+1)
# Observation = flat vector per task [count, max_wait, efficiency] + global stats

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
        # 0..num_needs-1: pick task, num_needs: VIP, num_needs+1: hold
        return spaces.Discrete(self.num_needs + 2)

    def _get_observation_space(self):
        obs_size = (self.num_needs * 3) + 4
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def _get_hold_action_number(self):
        return self.num_needs + 1

    def _get_invalid_action_reward(self):
        return -10.0

    def _get_customer_from_action(self, action) -> Customer:
        vip_action = self.num_needs
        wait_action = self.num_needs + 1
        unbearable_wait = self.limit_wait
        unbearable_appt = self.limit_vip_delay
        max_early = 60.0

        if action == wait_action:
            return None

        if action == vip_action:
            candidates = []
            for c in self.customer_waiting.values():
                if c.id in self.appointments:
                    if self.current_working_server.avg_service_time.get(c.task, 0) > 0:
                        appt_time = self.appointments[c.id].time
                        current_delay = self.system_time - appt_time
                        if current_delay < -max_early:
                            continue
                        if current_delay > unbearable_appt:
                            continue
                        candidates.append(c)
            if not candidates:
                return None
            return min(candidates, key=lambda c: self.appointments[c.id].time)

        task_id = action
        candidates = []
        for c in self.customer_waiting.values():
            if c.task == task_id and c.id not in self.appointments:
                current_wait = self.system_time - c.arrival_time
                if current_wait < unbearable_wait:
                    candidates.append(c)
        if not candidates:
            return None
        return min(candidates, key=lambda c: c.arrival_time)

    def _get_obs(self):
        waiting_customers, appointments, servers, expected_end, selected_server_id, sim_time = self._get_state()
        server = servers[selected_server_id]
        w_max = self.scenario.unbearable_wait if hasattr(self, 'scenario') else 60.0
        max_q = 50.0

        tasks_count = np.zeros(self.num_needs, dtype=np.float32)
        tasks_max_wait = np.zeros(self.num_needs, dtype=np.float32)
        my_efficiency = np.zeros(self.num_needs, dtype=np.float32)

        for c in waiting_customers.values():
            if c.id not in appointments:
                t_id = c.task
                tasks_count[t_id] += 1
                wait_t = sim_time - c.arrival_time
                if wait_t > tasks_max_wait[t_id]:
                    tasks_max_wait[t_id] = wait_t

        for t_id in range(self.num_needs):
            dur = server.avg_service_time.get(t_id, 0.0)
            my_efficiency[t_id] = dur / 60.0 if dur > 0 else 2.0

        vip_count = 0
        vip_urgency = 0.0
        for c in waiting_customers.values():
            if c.id in appointments:
                vip_count += 1
                delta = sim_time - appointments[c.id].time
                if delta > vip_urgency:
                    vip_urgency = delta

        tasks_count = np.clip(tasks_count / max_q, 0, 1)
        tasks_max_wait = np.clip(tasks_max_wait / w_max, 0, 2)
        norm_vip_count = np.clip(vip_count / max_q, 0, 1)
        norm_vip_urgency = np.clip(vip_urgency / 60.0, -1, 2)
        norm_time = sim_time / self.max_sim_time
        norm_id = float(selected_server_id) / self.c

        obs = []
        for t in range(self.num_needs):
            obs.extend([tasks_count[t], tasks_max_wait[t], my_efficiency[t]])
        obs.extend([norm_vip_count, norm_vip_urgency, norm_time, norm_id])
        return np.array(obs, dtype=np.float32)

    def action_masks(self):
        mask = [False] * (self.num_needs + 2)
        server = self.current_working_server
        if server is None:
            return mask
        waiting_customers = self.customer_waiting
        vip_action = self.num_needs
        wait_action = self.num_needs + 1
        limit_wait = self.limit_wait
        limit_vip_late = self.limit_vip_delay
        limit_vip_early = 60.0

        for t_id in range(self.num_needs):
            if server.avg_service_time.get(t_id, 0) > 0:
                for c in waiting_customers.values():
                    if c.task == t_id and c.id not in self.appointments:
                        if (self.system_time - c.arrival_time) < limit_wait:
                            mask[t_id] = True
                            break

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

        mask[wait_action] = not any(mask[:-1])
        return mask

    def _get_valid_reward(self, customer: Customer) -> float:
        reward = 20.0
        if customer.id in self.appointments:
            appt = self.appointments[customer.id]
            delta = abs(self.system_time - appt.time)
            reward += 50.0 if delta <= 15.0 else 10.0
        duration = self.current_working_server.avg_service_time.get(customer.task, 100.0)
        if duration > 0:
            reward += 300.0 / duration
        return reward
