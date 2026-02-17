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

    # =========================================================================
    # 1. ESPACES D'ACTION ULTRA-SIMPLIFIÉ
    # =========================================================================

    def _get_action_space(self):
        # 0 : Priorité aux VIP (Rendez-vous)
        # 1 : Priorité aux Walk-ins (Sans RDV)
        # 2 : Wait
        return spaces.Discrete(3)

    def _get_observation_space(self):
        # On garde l'observation détaillée pour que l'agent sache QUOI choisir.
        # Pour chaque tâche : [Walkin_Count, Walkin_Max_Wait, VIP_Count, VIP_Urgency, Efficiency]
        # + Global : [Time, My_ID]
        feats_per_task = 5
        obs_size = (self.num_needs * feats_per_task) + 2
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def _get_hold_action_number(self):
        return 2 # L'index de l'action WAIT

    # =========================================================================
    # 2. DÉCODAGE INTELLIGENT (HEURISTIQUES)
    # =========================================================================

    def _get_customer_from_action(self, action):
        wait_action = self._get_hold_action_number()
        if action == wait_action: return None

        server = self.current_working_server
        
        # --- PARAMÈTRES ---
        limit_wait = self.limit_wait      
        limit_vip_delay = self.limit_vip_delay 
        limit_vip_early = self.limit_vip_early       
        epsilon = 3.0

        # --- ACTION 0 : SERVE_VIP (Optimisé ROI) ---
        if action == 0:
            candidates = []
            for c in self.customer_waiting.values():
                # Filtre de base (RDV + Compétence + Fenêtre)
                if c.id in self.appointments and server.avg_service_time.get(c.task, 0) > 0:
                    delay = self.system_time - self.appointments[c.id].time
                    if -limit_vip_early <= delay <= limit_vip_delay:
                        candidates.append(c)
            
            if not candidates: return None
            
            # FONCTION DE SCORE INTELLIGENTE
            def vip_roi(c):
                # 1. Calculer le Score Qualité (0-100)
                appt_time = self.appointments[c.id].time
                delay = self.system_time - appt_time
                quality = 0.0
                
                if abs(delay) <= epsilon: quality = 100.0
                elif delay < -epsilon: quality = 100 * (1 - (abs(delay)-epsilon)/(limit_vip_early-epsilon))
                else: quality = 100 * (1 - (delay-epsilon)/(limit_vip_delay-epsilon))
                
                quality = max(0.1, quality) 

                # 2. Récupérer la Durée Estimée (Coût)
                duration = server.avg_service_time.get(c.task, 10.0)
                duration = max(1.0, duration) 

                # 3. Calculer le ROI AVEC BOOST
                # ON AJOUTE UN MULTIPLICATEUR (x2.0)
                # Cela signifie : "Même si c'est long, c'est un RDV, c'est sacré."
                # Cela permet de passer devant les petits walk-ins très rentables.
                roi = (quality * 2.0) / duration
                
                return (roi, -appt_time)

            # On prend le candidat avec le meilleur ROI
            return max(candidates, key=vip_roi)

        # --- ACTION 1 : SERVE_WALKIN (Optimisé ROI) ---
        elif action == 1:
            candidates = []
            for c in self.customer_waiting.values():
                # Filtre de base (Pas RDV + Compétence + Pas expiré)
                if c.id not in self.appointments and server.avg_service_time.get(c.task, 0) > 0:
                    if (self.system_time - c.arrival_time) < limit_wait:
                        candidates.append(c)
            
            if not candidates: return None
            
            # FONCTION DE SCORE INTELLIGENTE
            def walkin_roi(c):
                # 1. Calculer le Score Qualité Actuel (0-100)
                wait_time = self.system_time - c.arrival_time
                quality = 100.0 * (1.0 - (wait_time / limit_wait))
                quality = max(0.1, quality)

                # 2. Récupérer la Durée Estimée
                duration = server.avg_service_time.get(c.task, 10.0)
                duration = max(1.0, duration)

                # 3. ROI
                roi = quality / duration
                
                # Tie-breaker : Si ROI égal, on prend le plus ancien (FIFO) pour purger
                # Note : On veut maximiser le ROI, donc max()
                # arrival_time petit = ancien. Donc on met -arrival_time pour que le max le prenne.
                return (roi, -c.arrival_time)

            return max(candidates, key=walkin_roi)

        else:
            return None

    def _get_invalid_action_reward(self):
        return -10.0

    # =========================================================================
    # 3. MASQUE D'ACTION (CRUCIAL)
    # =========================================================================

    def action_masks(self):
        # [VIP, WALKIN, WAIT]
        mask = [False, False, False]
        
        server = self.current_working_server
        if server is None: return mask
        
        limit_wait = self.limit_wait 
        limit_vip_late = self.limit_vip_delay 
        limit_vip_early = self.limit_vip_early 

        # --- MASQUE VIP (Action 0) ---
        # True s'il existe AU MOINS UN patient VIP compatible et dans les temps
        has_vip = False
        for c in self.customer_waiting.values():
            if c.id in self.appointments:
                # Vérif compétence serveur
                if server.avg_service_time.get(c.task, 0) > 0:
                    delay = self.system_time - self.appointments[c.id].time
                    # Vérif fenêtre (On resserre un peu l'avance pour pas qu'il prenne trop tôt)
                    if -15 <= delay <= limit_vip_late:
                        has_vip = True
                        break
        mask[0] = has_vip

        # --- MASQUE WALKIN (Action 1) ---
        # True s'il existe AU MOINS UN patient Walkin compatible
        has_walkin = False
        for c in self.customer_waiting.values():
            if c.id not in self.appointments:
                # Vérif compétence serveur
                if server.avg_service_time.get(c.task, 0) > 0:
                    if (self.system_time - c.arrival_time) < limit_wait:
                        has_walkin = True
                        break
        mask[1] = has_walkin

        # --- MASQUE WAIT (Action 2) ---
        # Autorisé seulement si on ne peut rien faire d'autre (Anti-Grève)
        if mask[0] or mask[1]:
            mask[2] = False
        else:
            mask[2] = True
            
        return mask

    # =========================================================================
    # 4. OBSERVATION & REWARD (On garde ce qui marche)
    # =========================================================================

    def _get_obs(self):
        # Exactement la même observation qu'avant.
        # L'agent a besoin de voir le détail des tâches pour décider "VIP ou Walkin ?"
        waiting_customers, appointments, servers, _, selected_server_id, sim_time = self._get_state()
        server = servers[selected_server_id]
        
        w_max = self.limit_wait
        max_q = 50.0 
        
        walkin_counts = np.zeros(self.num_needs, dtype=np.float32)
        walkin_max_wait = np.zeros(self.num_needs, dtype=np.float32)
        vip_counts = np.zeros(self.num_needs, dtype=np.float32)
        vip_urgency = np.zeros(self.num_needs, dtype=np.float32)
        efficiency = np.zeros(self.num_needs, dtype=np.float32)

        for c in waiting_customers.values():
            t_id = c.task
            if c.id in appointments:
                vip_counts[t_id] += 1
                delta = sim_time - appointments[c.id].time
                if delta > vip_urgency[t_id]: vip_urgency[t_id] = delta
            else:
                walkin_counts[t_id] += 1
                wait_t = sim_time - c.arrival_time
                if wait_t > walkin_max_wait[t_id]: walkin_max_wait[t_id] = wait_t

        for t_id in range(self.num_needs):
            dur = server.avg_service_time.get(t_id, 0.0)
            if dur > 0: efficiency[t_id] = dur / 60.0
            else: efficiency[t_id] = 2.0

        walkin_counts = np.clip(walkin_counts / max_q, 0, 1)
        walkin_max_wait = np.clip(walkin_max_wait / w_max, 0, 2)
        vip_counts = np.clip(vip_counts / max_q, 0, 1)
        vip_urgency = np.clip(vip_urgency / 60.0, -1, 2) 

        obs = []
        for t in range(self.num_needs):
            obs.extend([walkin_counts[t], walkin_max_wait[t], vip_counts[t], vip_urgency[t], efficiency[t]])
        obs.extend([sim_time / self.max_sim_time, float(selected_server_id) / self.c])
        
        return np.array(obs, dtype=np.float32)

    def _get_valid_reward(self, customer: Customer) -> float:
        # On garde le bonus de survie + qualité
        quality_score = 0.0
        
        if customer.id in self.appointments:
            appt_time = self.appointments[customer.id].time
            delay = self.system_time - appt_time 
            epsilon = 3.0
            max_early = self.limit_vip_early
            max_late = self.limit_vip_delay
            
            if abs(delay) <= epsilon: quality_score = 100.0
            elif delay < -epsilon: quality_score = 100 * (1 - (abs(delay) - epsilon) / (max_early - epsilon))
            else: quality_score = 100 * (1 - (delay - epsilon) / (max_late - epsilon))
            quality_score = max(0.0, min(100.0, quality_score))

        else:
            wait_time = self.system_time - customer.arrival_time
            max_wait = self.limit_wait
            if wait_time < max_wait: quality_score = 100.0 * (1.0 - (wait_time / max_wait))
            else: quality_score = 0.0

        # BONUS THROUGHPUT (Vital)
        base_survival_bonus = 100.0 
        
        return base_survival_bonus + quality_score