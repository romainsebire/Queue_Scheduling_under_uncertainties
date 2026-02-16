from app.simulation.envs.Env import Env
from app.domain.Customer import Customer
from gymnasium import spaces
import numpy as np

class ChildEnv(Env):
    def __init__(self, mode, instance=None, scenario=None):
        # 1. DÉFINITION DES PARAMÈTRES (AVANT le parent)
        # On doit le faire ici car le parent va appeler reset() -> action_masks()
        # avant de nous rendre la main.
        
        if scenario:
            self.limit_wait = scenario.unbearable_wait
            # On utilise getattr pour éviter un crash si la variable change de nom dans le json
            self.limit_vip_delay = getattr(scenario, 'unbearable_wait_appointement', 30.0)
        else:
            self.limit_wait = 60.0
            self.limit_vip_delay = 30.0
            
        # On définit aussi la limite d'avance (hardcodée ou via config si dispo)
        self.limit_vip_early = 60.0

        # 2. INITIALISATION DU PARENT
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
        vip_action = self.num_needs
        wait_action = self.num_needs + 1
        
        # Paramètres exacts (Hard-coded pour correspondre à PolicyEvaluation)
        unbearable_wait = self.limit_wait      # 60
        unbearable_appt = self.limit_vip_delay # 30
        max_early = 60.0
        epsilon = 3.0 # Le plateau de tolérance
        
        # CAS 1 : WAIT
        if action == wait_action:
            return None

        # CAS 2 : VIP (Rendez-vous)
        if action == vip_action:
            candidates = []
            for c in self.customer_waiting.values():
                if c.id in self.appointments:
                    # 1. Compétence
                    if self.current_working_server.avg_service_time.get(c.task, 0) > 0:
                        
                        appt_time = self.appointments[c.id].time
                        delay = self.system_time - appt_time

                        # 2. FILTRE ZOMBIE (On ne prend que les 'sauvables')
                        # Doit être entre -60 (trop tôt) et +30 (trop tard)
                        if -max_early <= delay <= unbearable_appt:
                            candidates.append(c)
            
            if not candidates: return None
            
            # --- FONCTION DE SCORE LOCALE ---
            # On reproduit exactement la logique de PolicyEvaluation
            def potential_vip_score(c):
                appt_time = self.appointments[c.id].time
                # delay positif = retard, négatif = avance
                delay = self.system_time - appt_time 
                
                # Zone parfaite [-3, +3] -> 100 pts
                if abs(delay) <= epsilon:
                    return 100.0
                
                # Zone Avance [-60, -3[ -> Croissance linéaire
                elif delay < -epsilon:
                    # Si delay = -60 -> 0 pts. Si delay = -3 -> 100 pts.
                    # Formule simplifiée (proche de PolicyEvaluation)
                    score = 100 * (1 - (abs(delay) - epsilon) / (max_early - epsilon))
                    return max(0.0, score)
                
                # Zone Retard ]+3, +30] -> Décroissance linéaire (pente raide)
                else: # delay > epsilon
                    # Si delay = 3 -> 100 pts. Si delay = 30 -> 0 pts.
                    score = 100 * (1 - (delay - epsilon) / (unbearable_appt - epsilon))
                    return max(0.0, score)

            # TRI GREEDY : On prend celui qui a le MEILLEUR score MAINTENANT
            # Cela va favoriser naturellement :
            # 1. Ceux dans la zone +/- 3 min (Score 100)
            # 2. Ceux en avance (car la pente est douce, le score reste haut longtemps)
            # 3. Ceux en léger retard
            return max(candidates, key=potential_vip_score)

        # CAS 3 : TÂCHE WALK-IN
        task_id = action
        candidates = []
        for c in self.customer_waiting.values():
            if c.task == task_id and c.id not in self.appointments:
                
                # FILTRE ZOMBIE (Start Time)
                current_wait = self.system_time - c.arrival_time
                if current_wait < unbearable_wait:
                    candidates.append(c)
        
        if not candidates: return None
        
        # TRI GREEDY WALK-IN (LIFO)
        # On maximise le score courant : Score = 100 * (1 - wait/60)
        # Donc on veut minimiser le wait.
        # Donc on prend le plus récent (Arrival Time le plus grand).
        return max(candidates, key=lambda c: c.arrival_time)

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
        # 0..N-1 : Tâches Walk-in
        # N      : VIP
        # N+1    : WAIT
        mask = [False] * (self.num_needs + 2)
        
        server = self.current_working_server
        if server is None: return mask
        
        waiting_customers = self.customer_waiting
        vip_action = self.num_needs
        wait_action = self.num_needs + 1
        
        # --- Paramètres de tri (alignés avec PolicyEvaluation) ---
        # Limite Walk-in : 60 min d'attente max
        limit_wait = self.limit_wait 
        
        # Limite VIP Retard : 30 min max après l'heure
        limit_vip_late = self.limit_vip_delay 
        
        # Limite VIP Avance : On ne prend pas plus de 60 min avant l'heure
        limit_vip_early = 60.0 
        
        # --- 1. TÂCHES WALK-IN (0 à N-1) ---
        for t_id in range(self.num_needs):
            # A. Est-ce que je sais faire cette tâche ?
            if server.avg_service_time.get(t_id, 0) > 0:
                
                # B. Y a-t-il un candidat VIVANT pour cette tâche ?
                for c in waiting_customers.values():
                    if c.task == t_id and c.id not in self.appointments:
                        
                        # FILTRE "START TIME" :
                        # Le patient est-il en dessous de la limite d'attente MAINTENANT ?
                        current_wait = self.system_time - c.arrival_time
                        
                        if current_wait < limit_wait:
                            mask[t_id] = True
                            break # On en a trouvé au moins un, l'action est valide
                            
        # --- 2. VIP (Action N) ---
        has_vip = False
        for c in waiting_customers.values():
            if c.id in self.appointments:
                # A. Est-ce que je sais faire sa tâche ?
                if server.avg_service_time.get(c.task, 0) > 0:
                    
                    # B. Est-il dans la FENÊTRE DE TIR ?
                    appt_time = self.appointments[c.id].time
                    current_delay = self.system_time - appt_time
                    
                    # Condition 1 : Pas trop en avance (ex: pas avant -60 min)
                    # Condition 2 : Pas trop en retard (ex: pas après +30 min)
                    if -limit_vip_early <= current_delay <= limit_vip_late:
                        has_vip = True
                        break # On en a trouvé au moins un valide
                        
        if has_vip:
            mask[vip_action] = True
            
        # --- 3. WAIT (Action N+1) ---
        # Stratégie Anti-Grève : Si on peut travailler, on travaille.
        # On n'autorise WAIT que si absolument aucune autre action n'est possible.
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