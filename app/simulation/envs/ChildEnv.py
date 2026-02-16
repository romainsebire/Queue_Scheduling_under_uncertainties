from app.simulation.envs.Env import Env
from app.domain.Customer import Customer
from gymnasium import spaces
import numpy as np

class ChildEnv(Env):
    def __init__(self, mode, instance=None, scenario=None):
        # On définit les index d'action AVANT d'appeler super().__init__
        # car le parent va appeler _get_action_space immédiatement.
        
        # Note: self.num_needs n'est pas encore dispo ici si on n'a pas l'instance,
        # mais le parent stocke l'instance avant d'appeler les méthodes abstraites.
        # On fait confiance à l'architecture du parent.
        super().__init__(mode, instance, scenario)

    # =========================================================================
    # 1. MÉTHODES OBLIGATOIRES (ABSTRACT METHODS)
    # =========================================================================

    def _get_action_space(self):
        # Actions : 0 à num_needs-1 (Tâches), num_needs (VIP), num_needs+1 (WAIT)
        return spaces.Discrete(self.num_needs + 2)

    def _get_observation_space(self):
        # Taille = (num_needs * 3) + 4
        # [Count, Max_Wait, Efficiency] pour chaque tâche
        # + [VIP_Count, VIP_Urgency, Time, My_ID]
        obs_size = (self.num_needs * 3) + 4
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def _get_hold_action_number(self):
        # L'action "WAIT" est la dernière
        return self.num_needs + 1

    def _get_invalid_action_reward(self):
        # Pénalité si l'agent essaie une action impossible (filtrée par le masque normalement)
        return -10.0

    def _get_customer_from_action(self, action):
        """
        C'est ICI que réside l'intelligence "Chef de Service".
        L'agent donne une CATÉGORIE (action), nous trouvons le PATIENT.
        """
        vip_action = self.num_needs
        wait_action = self.num_needs + 1
        
        # CAS 1 : WAIT
        if action == wait_action:
            return None # Le parent gérera le _step_wait

        # CAS 2 : VIP (Rendez-vous)
        if action == vip_action:
            candidates = []
            for c in self.customer_waiting.values():
                if c.id in self.appointments:
                    # Vérification compétence serveur
                    if self.current_working_server.avg_service_time.get(c.task, 0) > 0:
                        candidates.append(c)
            
            if not candidates: return None
            # Tri : Le RDV le plus prioritaire (heure la plus petite)
            return min(candidates, key=lambda c: self.appointments[c.id].time)

        # CAS 3 : TÂCHE SPÉCIFIQUE (Walk-in)
        # L'action correspond à l'ID de la tâche (0, 1, 2...)
        task_id = action
        candidates = []
        for c in self.customer_waiting.values():
            if c.task == task_id and c.id not in self.appointments:
                candidates.append(c)
        
        if not candidates: return None
        # Tri : FIFO Strict (le plus vieux d'abord)
        return min(candidates, key=lambda c: c.arrival_time)

    # =========================================================================
    # 2. LOGIQUE D'OBSERVATION (TABLEAU DE BORD)
    # =========================================================================

    def _get_obs(self):
        waiting_customers, appointments, servers, expected_end, selected_server_id, sim_time = self._get_state()
        server = servers[selected_server_id]
        
        # Normalisation
        w_max = self.scenario.unbearable_wait if hasattr(self, 'scenario') else 60.0
        max_q = 50.0 
        
        # Init Vecteurs
        tasks_count = np.zeros(self.num_needs, dtype=np.float32)
        tasks_max_wait = np.zeros(self.num_needs, dtype=np.float32)
        my_efficiency = np.zeros(self.num_needs, dtype=np.float32)
        
        # Stats Walk-ins
        for c in waiting_customers.values():
            if c.id not in appointments:
                t_id = c.task
                tasks_count[t_id] += 1
                wait_t = sim_time - c.arrival_time
                if wait_t > tasks_max_wait[t_id]:
                    tasks_max_wait[t_id] = wait_t

        # Stats Serveur (Moi)
        for t_id in range(self.num_needs):
            dur = server.avg_service_time.get(t_id, 0.0)
            if dur > 0:
                my_efficiency[t_id] = dur / 60.0
            else:
                my_efficiency[t_id] = 2.0 # Incompétent
        
        # Stats VIP
        vip_count = 0
        vip_urgency = 0.0
        for c in waiting_customers.values():
            if c.id in appointments:
                vip_count += 1
                delta = sim_time - appointments[c.id].time
                if delta > vip_urgency:
                    vip_urgency = delta
        
        # Normalisation finale
        tasks_count = np.clip(tasks_count / max_q, 0, 1)
        tasks_max_wait = np.clip(tasks_max_wait / w_max, 0, 2)
        norm_vip_count = np.clip(vip_count / max_q, 0, 1)
        norm_vip_urgency = np.clip(vip_urgency / 60.0, -1, 2)
        norm_time = sim_time / self.max_sim_time
        norm_id = float(selected_server_id) / self.c
        
        # Assemblage
        obs = []
        for t in range(self.num_needs):
            obs.extend([tasks_count[t], tasks_max_wait[t], my_efficiency[t]])
        obs.extend([norm_vip_count, norm_vip_urgency, norm_time, norm_id])
        
        return np.array(obs, dtype=np.float32)

    # =========================================================================
    # 3. MASQUE & REWARD
    # =========================================================================

    def action_masks(self):
        mask = [False] * (self.num_needs + 2)
        server = self.current_working_server
        if server is None: return mask
        
        waiting_customers = self.customer_waiting
        vip_action = self.num_needs
        wait_action = self.num_needs + 1
        
        # 1. Tâches Walk-in
        for t_id in range(self.num_needs):
            if server.avg_service_time.get(t_id, 0) > 0:
                # Y a-t-il un candidat ?
                for c in waiting_customers.values():
                    if c.task == t_id and c.id not in self.appointments:
                        mask[t_id] = True
                        break

        # 2. VIP
        has_vip = False
        for c in waiting_customers.values():
            if c.id in self.appointments:
                if server.avg_service_time.get(c.task, 0) > 0:
                    has_vip = True
                    break
        if has_vip:
            mask[vip_action] = True
            
        # 3. Wait
        # Si on peut travailler, on travaille (Anti-Grève)
        if any(mask[:-1]): 
            mask[wait_action] = False
        else:
            mask[wait_action] = True
            
        return mask

    def _get_valid_reward(self, customer: Customer) -> float:
        # Récompense Stratégique
        reward = 20.0 # Base (Throughput)
        
        # VIP
        if customer.id in self.appointments:
            appt = self.appointments[customer.id]
            delta = abs(self.system_time - appt.time)
            if delta <= 15.0: reward += 50.0
            else: reward += 10.0
        
        # Spécialiste (Vitesse)
        duration = self.current_working_server.avg_service_time.get(customer.task, 100.0)
        if duration > 0:
            reward += (300.0 / duration)
            
        return reward