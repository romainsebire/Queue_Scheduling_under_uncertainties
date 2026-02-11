import numpy as np
import gymnasium as gym
from gymnasium import spaces
from app.simulation.envs.Env import Env
from app.domain.Customer import Customer

class ChildEnv(Env):
    def __init__(self, mode, instance=None, scenario=None):
        # Sizing parameters for RL
        self.max_obs_queue = 100
        self.num_server_features = 2  # [Busy?, Time_until_free] + avg_service_times
        self.num_queue_features = 8   # [WaitTime, TaskID, IsAppt?, ApptDelta, EstServiceTime, AbandonTime] + Fit Score + Skill Score
        
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
        
        # --- 1. LOGIQUE DE TRI (STRATÉGIE GAGNANTE : VIP + FIFO) ---
        def priority_key(c):
            # GROUPE 1 : LES RENDEZ-VOUS (Priorité Absolue)
            if c.id in appointments:
                # On les met tout en haut (-1 milliard) triés par heure prévue
                return -1000000000 + appointments[c.id].time
            
            # GROUPE 2 : LES WALK-INS (FIFO)
            # Ordre d'arrivée strict pour maximiser le score Waiting Time
            return c.arrival_time

        sorted_customers = sorted(waiting_customers.values(), key=priority_key)
        
        # Mise à jour du mapping (L'agent agit sur ces IDs là)
        self.queue_mapping = [c.id for c in sorted_customers[:self.max_obs_queue]]
        
        # --- 2. CALCUL DU TEMPS DISPONIBLE (POUR LE FIT SCORE) ---
        # On cherche la prochaine pause ou la fin de journée pour CE serveur
        time_limit = self.max_sim_time
        next_break_start = float('inf')
        
        # On scanne les activités planifiées (pauses futures)
        for activity in self.planned_server_activity.values():
            if activity.server_id == selected_server_id:
                if activity.expected_start < next_break_start:
                    next_break_start = activity.expected_start
        
        # Le temps réel dont dispose le serveur maintenant
        real_time_available = min(time_limit, next_break_start) - sim_time

        # --- 3. CONSTRUCTION DE LA MATRICE PATIENTS ---
        queue_matrix = np.zeros((self.max_obs_queue, self.num_queue_features), dtype=np.float32)
        
        # Récupération du W_max pour normaliser l'attente
        w_max = self.scenario.unbearable_wait if hasattr(self, 'scenario') else 60.0
        if w_max == 0: w_max = 60.0

        current_server_obj = servers[selected_server_id]

        for i, customer in enumerate(sorted_customers):
            if i >= self.max_obs_queue: break
            
            # --- Features Standards (0-5) ---
            
            # 0. Temps d'attente (Normalisé)
            wait_time = (sim_time - customer.arrival_time) / w_max
            
            # 1. Task ID (Normalisé)
            norm_task = float(customer.task) / self.num_needs
            
            # 2. Is Appointment (Binaire)
            is_appt = 1.0 if customer.id in appointments else 0.0
            
            # 3. Appointment Delta (Normalisé)
            appt_delta = 0.0
            if is_appt:
                raw_delta = sim_time - appointments[customer.id].time
                appt_delta = np.clip(raw_delta / 60.0, -1.0, 1.0)
            
            # 4. Estimated Service Time (Global Average Normalisé)
            # On prend une moyenne globale pour que l'agent ait une idée de la lourdeur de la tâche
            # (Indépendamment du serveur actuel)
            est_service = 15.0 # Valeur par défaut
            times = [s.avg_service_time.get(customer.task, 0) for s in servers.values()]
            valid_times = [t for t in times if t > 0]
            if valid_times:
                est_service = sum(valid_times) / len(valid_times)
            norm_est_service = np.clip(est_service / 60.0, 0.0, 1.0)
            
            # 5. Abandonment Time (Normalisé - Urgence)
            time_before_abandon = 1.0 # 1.0 = Pas urgent
            if customer.abandonment_time is not None:
                raw_left = customer.abandonment_time - sim_time
                # 0.0 = Très urgent (va partir), 1.0 = Large
                time_before_abandon = np.clip(raw_left / w_max, 0.0, 1.0)

            # --- Features Intelligentes (6-7) ---
            
            # 6. SKILL SCORE : Ce serveur est-il rapide pour ça ?
            # On compare le temps de CE serveur par rapport à 60min (ou à la moyenne)
            my_time = current_server_obj.avg_service_time.get(customer.task, 999.0)
            # Si my_time est petit (ex: 10min), score proche de 1.
            # Si my_time est grand (ex: 60min), score proche de 0.
            skill_score = 1.0 - np.clip(my_time / 60.0, 0.0, 1.0)
            if my_time == 0: skill_score = 0.0 # Incompétent

            # 7. FIT SCORE : Est-ce que ça rentre avant la pause ? (Tetris)
            # 1.0 = Ça rentre large
            # 0.5 = Ça rentre tout juste
            # 0.0 = Ça ne rentre pas (Danger !)
            fit_score = 0.0
            if my_time > 0 and real_time_available > 0:
                if my_time <= real_time_available:
                     # Plus on a de marge, plus le score est haut (optionnel, ou juste 1.0)
                     fit_score = 1.0 
                else:
                     fit_score = 0.0 # Ne rentre pas

            queue_matrix[i] = [
                wait_time, 
                norm_task, 
                is_appt, 
                appt_delta, 
                norm_est_service,
                time_before_abandon,
                skill_score,  # NEW
                fit_score     # NEW
            ]

        # --- 4. CONSTRUCTION MATRICE SERVEURS ---
        # On garde la vision globale des serveurs
        server_matrix = np.zeros((self.c, 2 + self.num_needs), dtype=np.float32)
        for s_id in range(self.c):
            is_busy = 1.0 if expected_end[s_id] > sim_time else 0.0
            remaining = max(0.0, expected_end[s_id] - sim_time) / 60.0
            
            server_matrix[s_id, 0] = is_busy
            server_matrix[s_id, 1] = remaining
            
            for task_id in range(self.num_needs):
                val = servers[s_id].avg_service_time.get(task_id, 0.0)
                server_matrix[s_id, 2 + task_id] = val / 60.0

        # --- 5. CONTEXTE ---
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
        server = self.current_working_server
        if server is None: return mask
        
        # Combien de temps avant la fin de la journée (ou pause)
        # Simplification : on regarde juste la fin de journée
        time_available = self.max_sim_time - self.system_time

        can_serve_someone = False
        
        for i in range(len(self.queue_mapping)):
            cust_id = self.queue_mapping[i]
            customer = self.customer_waiting.get(cust_id)
            if customer:
                # 1. Compétence
                task_duration = server.avg_service_time.get(customer.task, 0)
                if task_duration > 0:
                    # 2. TEMPS DISPONIBLE (Le Tetris)
                    # On accepte seulement si on a le temps de finir (avec marge de sécurité)
                    if task_duration <= time_available:
                        mask[i] = True
                        can_serve_someone = True
        
        # Gestion du HOLD
        if can_serve_someone:
            mask[self._get_hold_action_number()] = False
        else:
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