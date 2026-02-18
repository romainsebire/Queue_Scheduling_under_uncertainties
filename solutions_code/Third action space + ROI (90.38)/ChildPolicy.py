import os
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from app.simulation.policies.Policy import Policy
from app.simulation.envs.Env import Env

class ChildPolicy(Policy):
    def __init__(self, model_title):
        super().__init__(model_title)
        self.model = None
        # path to save trained model
        self.model_path = os.path.join("app", "data", "models", "ppo_child_policy")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    @staticmethod
    def _mask_fn(env):
        """
        helper to extract action mask from environment
        """
        return env.unwrapped.action_masks()

    def learn(self, scenario, total_timesteps, verbose=1, callback=None, **kwargs):
        """
        trains the rl agent using maskableppo
        """
        print(f"--- Starting Training for {total_timesteps} steps ---")
        
        # create training environment
        env = gym.make("Child_Env", mode=Env.MODE.TRAIN, scenario=scenario)
        # monitor wrapper for logging
        env = Monitor(env)
        
        # wrapper to handle invalid actions
        env = ActionMasker(env, self._mask_fn)

        # initialize maskable ppo model
        self.model = MaskablePPO(
            "MlpPolicy",
            env, 
            verbose=1,
            tensorboard_log="app/data/logs", 
            learning_rate=3e-4,
            batch_size=256,
            n_steps=2048,   
            gamma=0.99,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256])
        )

        # start training
        self.model.learn(total_timesteps=total_timesteps, callback=callback, **kwargs)
        
        # save model
        self.model.save(self.model_path)
        print(f"--- Training Finished. Model saved at {self.model_path}.zip ---")

    def _predict(self, obs, info):
        """
        used during simulation phase
        """
        # load model if needed
        if self.model is None:
            if os.path.exists(self.model_path + ".zip"):
                self.model = MaskablePPO.load(self.model_path)
            else:
                # fallback: random valid action if model missing
                mask = info.get("action_mask")
                valid_actions = [i for i, valid in enumerate(mask) if valid]
                return np.random.choice(valid_actions)

        # retrieve action mask
        action_masks = np.array(info.get("action_mask"), dtype=bool)

        # predict deterministic action
        action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
        
        return int(action)