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
        waiting_customers, appointments, servers, expected_end, selected_server_id, sim_time = self._get_state()
        
        # --- STRATÉGIE DE TRI HEURISTIQUE ---
        def priority_key(c):
            # --- GROUPE 1 : LES RENDEZ-VOUS (Priorité Absolue) ---
            if c.id in appointments:
                # -1 milliard pour être sûr qu'ils sont devant
                return -1000000000 + appointments[c.id].time
            
            # --- GROUPE 2 : LES WALK-INS (FIFO) ---
            # Ordre d'arrivée strict. C'est ce qui garantit le score Waiting Time.
            return c.arrival_time

        # Application du tri
        sorted_customers = sorted(waiting_customers.values(), key=priority_key)
        
        # Mise à jour du mapping (L'agent verra les VIPs aux actions 0, 1, 2...)
        self.queue_mapping = [c.id for c in sorted_customers[:self.max_obs_queue]]
        
        queue_matrix = np.zeros((self.max_obs_queue, self.num_queue_features), dtype=np.float32)
        
        # On récupère le W_max pour normaliser
        w_max = self.scenario.unbearable_wait if hasattr(self, 'scenario') else 60.0
        if w_max == 0: w_max = 60.0

        for i, customer in enumerate(sorted_customers):
            if i >= self.max_obs_queue: break
            
            # --- NORMALISATION ---
            # Wait time entre 0 et ~1 (parfois >1 si on dépasse)
            wait_time = (sim_time - customer.arrival_time) / w_max
            
            # Appt delta entre -1 et 1 environ
            appt_delta = 0.0
            is_appt = 0.0
            if customer.id in appointments:
                is_appt = 1.0
                raw_delta = sim_time - appointments[customer.id].time
                appt_delta = np.clip(raw_delta / 60.0, -1.0, 1.0) # On clip pour éviter les valeurs folles
            
            # Est Service time normalisé (divisé par ex par 30 min)
            avg_duration = 10.0 # Fallback
            # On essaie d'être précis sur l'estimation
            # On prend la moyenne des durées possibles pour cette tâche sur TOUS les serveurs
            potential_times = [s.avg_service_time.get(customer.task, 0) for s in servers.values()]
            valid_times = [t for t in potential_times if t > 0]
            if valid_times:
                avg_duration = sum(valid_times) / len(valid_times)
            est_service = np.clip(avg_duration / 30.0, 0.0, 2.0)

            # Abandon time normalisé
            time_before_abandon = 1.0 # Valeur par défaut (loin)
            if customer.abandonment_time is not None:
                # 1.0 = c'est urgent (reste 0 temps), 0.0 = on a le temps (reste w_max)
                raw_left = customer.abandonment_time - sim_time
                time_before_abandon = 1.0 - np.clip(raw_left / w_max, 0.0, 1.0)

            queue_matrix[i] = [
                wait_time, 
                float(customer.task) / self.num_needs, # On normalise aussi l'ID de tache
                is_appt, 
                appt_delta, 
                est_service,
                time_before_abandon
            ]

        # --- SERVEURS ---
        server_matrix = np.zeros((self.c, 2 + self.num_needs), dtype=np.float32)
        for s_id in range(self.c):
            is_busy = 1.0 if expected_end[s_id] > sim_time else 0.0
            remaining = max(0.0, expected_end[s_id] - sim_time) / 30.0 # Normalisé
            
            server_matrix[s_id, 0] = is_busy
            server_matrix[s_id, 1] = remaining
            
            # Capacités normalisées
            for task_id in range(self.num_needs):
                val = servers[s_id].avg_service_time.get(task_id, 0.0)
                server_matrix[s_id, 2 + task_id] = val / 30.0

        # --- CONTEXTE ---
        # On ajoute le ID du serveur sélectionné, mais sous forme One-Hot ou normalisée
        # Ici normalisée simple
        context = np.array([sim_time / self.max_sim_time, float(selected_server_id) / self.c], dtype=np.float32)

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
        mask = [False] * (self.max_obs_queue + 1)
        current_server = self.current_working_server
        
        if current_server is None: return mask 

        num_customers = len(self.queue_mapping)
        can_serve_someone = False
        
        for i in range(num_customers):
            cust_id = self.queue_mapping[i]
            customer = self.customer_waiting.get(cust_id)
            
            if customer:
                # Vérification basique de compétence (temps moyen > 0)
                if current_server.avg_service_time.get(customer.task, 0) > 0:
                    mask[i] = True
                    can_serve_someone = True
        
        # --- MODIFICATION CRUCIALE ---
        # Si on peut servir au moins une personne, INTERDICTION de faire HOLD.
        if can_serve_someone:
            mask[self._get_hold_action_number()] = False
        else:
            # Si vraiment personne n'est compatible, alors (et seulement alors) on peut attendre
            mask[self._get_hold_action_number()] = True
        
        return mask

    def _get_invalid_action_reward(self) -> float: 
        """
        The agent tried to serve a ghost (empty index).
        """
        return -10.0
    
    def _get_valid_reward(self, customer: Customer) -> float:
        reward = 0.0
        
        # 1. Base (Throughput)
        # On garde +15. C'est assez fort pour interdire le HOLD, 
        # mais assez faible pour ne pas masquer les bonus RDV.
        reward += 15.0
        
        # 2. Bonus Rendez-vous (VIP)
        if customer.id in self.appointments:
            appt = self.appointments[customer.id]
            delta = abs(self.system_time - appt.time)
            
            if delta <= 5.0:       
                reward += 40.0  # Total = 55
            elif delta <= 30.0:    
                reward += 15.0  # Total = 30
            else:                  
                reward += 0.0   # Total = 15
        
        # 3. Bonus Walk-in (Gestion FIFO)
        else:
            w_max = self.scenario.unbearable_wait if hasattr(self, 'scenario') else 60.0
            wait_ratio = (self.system_time - customer.arrival_time) / w_max
            
            # On récompense le fait de traiter des patients qui sont encore dans la course
            if wait_ratio < 1.0:
                reward += 5.0 * (1.0 - wait_ratio)

        return reward