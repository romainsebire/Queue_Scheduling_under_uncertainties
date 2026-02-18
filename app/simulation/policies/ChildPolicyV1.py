# Policy for Solution V1 — Patient ID action space (score ~60%)
# Uses MultiInputPolicy because ChildEnvV1 has a Dict observation space

import os
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from app.simulation.policies.Policy import Policy
from app.simulation.envs.Env import Env


class ChildPolicy(Policy):
    def __init__(self, model_title, seed=42):
        super().__init__(model_title)
        self.model = None
        self.seed = seed
        self.model_path = os.path.join("app", "data", "models", "ppo_first")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    @staticmethod
    def _mask_fn(env):
        return env.unwrapped.action_masks()

    def learn(self, scenario, total_timesteps, verbose=1, callback=None, **kwargs):
        print(f"--- [V1] Starting Training for {total_timesteps} steps (seed={self.seed}) ---")

        env = gym.make("ChildEnv", mode=Env.MODE.TRAIN, scenario=scenario)
        env = Monitor(env)
        env = ActionMasker(env, self._mask_fn)

        self.model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=verbose,
            seed=self.seed,
            tensorboard_log=os.path.join("app", "data", "logs"),
            learning_rate=3e-3,
            gamma=0.99,
            ent_coef=0.01,
            batch_size=512,
            n_steps=2048,
            policy_kwargs=dict(net_arch=[256, 256]),
        )

        self.model.learn(total_timesteps=total_timesteps, callback=callback, **kwargs)
        self.model.save(self.model_path)
        print(f"--- [V1] Training Finished. Model saved at {self.model_path}.zip ---")

    def _predict(self, obs, info):
        if self.model is None:
            if os.path.exists(self.model_path + ".zip"):
                self.model = MaskablePPO.load(self.model_path)
            else:
                mask = info.get("action_mask")
                valid_actions = [i for i, valid in enumerate(mask) if valid]
                return np.random.choice(valid_actions)

        action_masks = np.array(info.get("action_mask"), dtype=bool)
        action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
        return int(action)
