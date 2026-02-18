from app.simulation.envs.Env import Env
from app.domain.Customer import Customer
import gymnasium as gym
import numpy as np

MAX_CUSTOMERS = 15    
MAX_WAIT = 60.0       # max acceptable waiting time for walk-ins (minutes)
MAX_EARLY = 60.0      # max early arrival for appointments
MAX_LATE = 30.0       # max late arrival for appointments
EPSILON_APPT = 3.0    # perfect appointment window (+-3 min)
MAX_SIM_TIME = 630.0  # 10.5 hours in minutes
MAX_SERVICE_TIME = 30.0  # max average service time from config

# ── Semantic action indices ───────────────────────────────────────────────────
#  The agent picks a scheduling strategy; the env deterministically resolves
#  it to the best matching customer.  This collapses a 16-action space into 4
#  highly interpretable actions that map directly onto the reward components.
N_ACTIONS = 4
ACTION_SERVE_APPT_URGENT  = 0   # serve most overdue/closest appointment customer
ACTION_SERVE_LONGEST_WAIT = 1   # serve longest-waiting servable customer
ACTION_SERVE_FASTEST      = 2   # serve lowest service-time servable customer
ACTION_HOLD               = 3   # do not serve, let simulation advance

# ── Observation layout ───────────────────────────────────────────────────────
#  For each of the 3 "serve" strategies we expose 5 features describing the
#  best candidate that would be chosen.  Slots are zeroed when no candidate
#  exists (i.e. the corresponding action is masked).
#
#  Indices 0-4   → best APPT_URGENT  candidate features
#  Indices 5-9   → best LONGEST_WAIT candidate features
#  Indices 10-14 → best FASTEST      candidate features
#  Index  15     → has_appt_candidate   (0/1)
#  Index  16     → has_any_servable     (0/1)
#  Index  17     → num_servable / MAX_CUSTOMERS
#  Indices 18-21 → global: sim_time, queue_len, server_id, server_available
FEATURES_PER_CANDIDATE = 5   # wait, service, has_appt, time_to_appt, urgency
OBS_SIZE = FEATURES_PER_CANDIDATE * 3 + 3 + 4   # = 22


class ChildEnv(Env):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_action_space(self):
        """4 semantic strategy actions (see module-level constants)."""
        return gym.spaces.Discrete(N_ACTIONS)

    def _get_observation_space(self):
        """22-dimensional normalized observation."""
        return gym.spaces.Box(
            low=-1.0,
            high=2.0,
            shape=(OBS_SIZE,),
            dtype=np.float32
        )


    def _get_obs(self):
        """
        Build compact 22-dim observation.

        Per-candidate block (5 features, repeated × 3 strategies):
          0: wait_time / MAX_WAIT
          1: service_time / MAX_SERVICE_TIME
          2: has_appointment  (0 or 1)
          3: (appt_time - sim_time) / MAX_WAIT  (negative = overdue)
          4: urgency  (0-1)

        Availability flags (3):
          15: has_appt_candidate  (1 if SERVE_APPT_URGENT is valid)
          16: has_any_servable    (1 if SERVE_LONGEST_WAIT / FASTEST valid)
          17: num_servable / MAX_CUSTOMERS

        Global (4):
          18: sim_time / MAX_SIM_TIME
          19: queue_length / MAX_CUSTOMERS
          20: server_id / num_servers
          21: server_is_available  (0 or 1)
        """
        _, appointments, servers, expected_end, selected_server_id, sim_time \
            = self._get_state()

        num_servers = self.c
        current_server = servers[selected_server_id]
        servable = self._get_servable_customers()

        def _customer_features(cid, customer):
            waiting_time = sim_time - customer.arrival_time
            has_appt = 1.0 if cid in appointments else 0.0
            if has_appt:
                time_to_appt = appointments[cid].time - sim_time
                urgency = float(np.clip(1.0 - time_to_appt / MAX_LATE, 0.0, 1.0))
            else:
                time_to_appt = MAX_SIM_TIME
                urgency = float(np.clip(waiting_time / MAX_WAIT, 0.0, 1.0))

            est_service = current_server.avg_service_time[customer.task]
            return [
                float(np.clip(waiting_time / MAX_WAIT, 0.0, 2.0)),
                float(np.clip(est_service / MAX_SERVICE_TIME, 0.0, 1.0)),
                has_appt,
                float(np.clip(time_to_appt / MAX_WAIT, -1.0, 2.0)),
                urgency,
            ]

        ZERO_BLOCK = [0.0] * FEATURES_PER_CANDIDATE

        # ── Strategy 0: SERVE_APPT_URGENT ────────────────────────────────────
        appt_servable = [(cid, c) for cid, c in servable if cid in appointments]
        if appt_servable:
            best_appt_cid, best_appt_c = min(
                appt_servable,
                key=lambda x: appointments[x[0]].time - sim_time   # most urgent
            )
            block0 = _customer_features(best_appt_cid, best_appt_c)
        else:
            block0 = ZERO_BLOCK

        # ── Strategy 1: SERVE_LONGEST_WAIT ───────────────────────────────────
        if servable:
            best_wait_cid, best_wait_c = max(
                servable, key=lambda x: sim_time - x[1].arrival_time
            )
            block1 = _customer_features(best_wait_cid, best_wait_c)
        else:
            block1 = ZERO_BLOCK

        # ── Strategy 2: SERVE_FASTEST ─────────────────────────────────────────
        if servable:
            best_fast_cid, best_fast_c = min(
                servable,
                key=lambda x: current_server.avg_service_time[x[1].task]
            )
            block2 = _customer_features(best_fast_cid, best_fast_c)
        else:
            block2 = ZERO_BLOCK

        # ── Availability flags ────────────────────────────────────────────────
        has_appt_candidate = 1.0 if appt_servable else 0.0
        has_any_servable   = 1.0 if servable else 0.0
        num_servable_norm  = float(np.clip(len(servable) / max(MAX_CUSTOMERS, 1), 0.0, 1.0))

        # ── Global ────────────────────────────────────────────────────────────
        global_feats = [
            sim_time / MAX_SIM_TIME,
            len(self.customer_waiting) / max(MAX_CUSTOMERS, 1),
            selected_server_id / max(num_servers - 1, 1),
            1.0 if expected_end[selected_server_id] == 0 else 0.0,
        ]

        obs = (block0 + block1 + block2
               + [has_appt_candidate, has_any_servable, num_servable_norm]
               + global_feats)
        return np.array(obs, dtype=np.float32)

    # ── Action resolution ─────────────────────────────────────────────────────

    def _get_servable_customers(self):
        """Return list of (cid, customer) the current server can handle."""
        return [
            (cid, c)
            for cid, c in self.customer_waiting.items()
            if self.current_working_server.avg_service_time[c.task] > 0
        ]

    def _get_customer_from_action(self, action) -> Customer:
        """Resolve semantic action to a concrete customer, or None (HOLD)."""
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)

        if action == ACTION_HOLD:
            return None

        _, appointments, servers, _, selected_server_id, sim_time = self._get_state()
        current_server = servers[selected_server_id]
        servable = self._get_servable_customers()

        if not servable:
            return None

        if action == ACTION_SERVE_APPT_URGENT:
            appt_servable = [(cid, c) for cid, c in servable if cid in appointments]
            if not appt_servable:
                return None
            _, customer = min(
                appt_servable,
                key=lambda x: appointments[x[0]].time - sim_time
            )
            return customer

        if action == ACTION_SERVE_LONGEST_WAIT:
            _, customer = max(
                servable,
                key=lambda x: sim_time - x[1].arrival_time
            )
            return customer

        if action == ACTION_SERVE_FASTEST:
            _, customer = min(
                servable,
                key=lambda x: current_server.avg_service_time[x[1].task]
            )
            return customer

        return None

    # ── Reward ────────────────────────────────────────────────────────────────

    def _get_invalid_action_reward(self) -> float:
        return -10.0

    def _get_valid_reward(self, customer: Customer) -> float:
        """
        Mirrors evaluation metrics:
          40% waiting time score + 40% appointment compliance + 20% throughput
        """
        waiting_time = self.system_time - customer.arrival_time

        if customer.id in self.appointments:
            appt_time = self.appointments[customer.id].time
            diff = self.system_time - appt_time

            if abs(diff) <= EPSILON_APPT:
                appt_score = 100.0
            elif diff < -EPSILON_APPT and diff > -MAX_EARLY:
                appt_score = 100.0 * (1 + (diff + EPSILON_APPT) / (MAX_EARLY - EPSILON_APPT))
            elif diff > EPSILON_APPT and diff < MAX_LATE:
                appt_score = 100.0 / (MAX_LATE - EPSILON_APPT) * (MAX_LATE - diff)
            else:
                appt_score = 0.0

            if waiting_time >= MAX_WAIT:
                wait_score = 0.0
            else:
                wait_score = 100.0 * (1.0 - waiting_time / MAX_WAIT)

            reward = 0.4 * wait_score + 0.4 * appt_score + 0.2 * 100.0
        else:
            if waiting_time >= MAX_WAIT:
                wait_score = 0.0
            else:
                wait_score = 100.0 * (1.0 - waiting_time / MAX_WAIT)

            reward = 0.8 * wait_score + 0.2 * 100.0

        return reward

    def _get_hold_reward(self) -> float:
        """Scale HOLD penalty with queue pressure."""
        queue_len = len(self.customer_waiting)
        if queue_len == 0:
            return 0.0
        return -2.0 * min(queue_len / MAX_CUSTOMERS, 1.0)

    # ── Action masking ────────────────────────────────────────────────────────

    def action_masks(self):
        """
        Boolean mask for the 4 semantic actions.
          SERVE_APPT_URGENT : valid iff ≥1 servable appointment customer
          SERVE_LONGEST_WAIT: valid iff ≥1 servable customer
          SERVE_FASTEST     : valid iff ≥1 servable customer
          HOLD              : always valid
        """
        servable = self._get_servable_customers()
        _, appointments, _, _, _, _ = self._get_state()
        has_appt = any(cid in appointments for cid, _ in servable)

        mask = [False] * N_ACTIONS
        mask[ACTION_SERVE_APPT_URGENT]  = has_appt
        mask[ACTION_SERVE_LONGEST_WAIT] = bool(servable)
        mask[ACTION_SERVE_FASTEST]      = bool(servable)
        mask[ACTION_HOLD]               = True
        return mask

    def _get_hold_action_number(self):
        return ACTION_HOLD
