from app.simulation.policies.Policy import Policy
from app.simulation.envs.Env import Env
import gymnasium as gym
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


class ChildPolicy(Policy):
    def __init__(self, model_title="ChildPolicy"):
        super().__init__(model_title)
        self.model = None

    def _predict(self, obs, info):
        if self.model is None:
            return self._heuristic_policy(obs, info)

        action_mask = np.array(info.get('action_mask', None))
        action, _states = self.model.predict(obs, action_masks=action_mask, deterministic=True)
        return action

    def _heuristic_policy(self, obs, info):
        """
        Simple heuristic over the 4 semantic actions:
          0 SERVE_APPT_URGENT  → prefer if any urgent appointment candidate exists
          1 SERVE_LONGEST_WAIT → prefer if any servable customer exists
          2 SERVE_FASTEST      → fallback serve action
          3 HOLD               → last resort
        """
        from app.simulation.envs.ChildEnv import (
            ACTION_SERVE_APPT_URGENT, ACTION_SERVE_LONGEST_WAIT,
            ACTION_SERVE_FASTEST, ACTION_HOLD,
        )
        action_mask = info.get('action_mask', [True, True, True, True])

        # obs layout: block0(appt) | block1(wait) | block2(fast) | flags | global
        # feature index 4 in each block = urgency
        appt_urgency = obs[4]          # urgency of best appointment candidate
        has_appt     = obs[15]         # flag: appointment candidate exists
        has_servable = obs[16]         # flag: any servable customer exists

        # Prefer urgent appointment service when the candidate is highly urgent
        if action_mask[ACTION_SERVE_APPT_URGENT] and has_appt and appt_urgency > 0.5:
            return ACTION_SERVE_APPT_URGENT

        # Otherwise serve the longest-waiting customer
        if action_mask[ACTION_SERVE_LONGEST_WAIT] and has_servable:
            return ACTION_SERVE_LONGEST_WAIT

        # Fallback: serve fastest
        if action_mask[ACTION_SERVE_FASTEST] and has_servable:
            return ACTION_SERVE_FASTEST

        return ACTION_HOLD

    def learn(self, scenario, total_timesteps, verbose):
        def mask_fn(env):
            return env.unwrapped.action_masks()

        learning_env = gym.make("ChildEnv", mode=Env.MODE.TRAIN, scenario=scenario)
        learning_env = ActionMasker(learning_env, mask_fn)

        print(f"Starting training for {total_timesteps} timesteps...")

        # Custom network: separate policy and value networks [128, 64]
        policy_kwargs = dict(
            net_arch=dict(pi=[128, 64], vf=[128, 64])
        )

        self.model = MaskablePPO(
            "MlpPolicy",
            learning_env,
            verbose=1 if verbose else 0,
            device="cpu",
            learning_rate=lambda p: 3e-4 * p,
            n_steps=4096,
            batch_size=128,
            n_epochs=5,      
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            policy_kwargs=policy_kwargs,
            tensorboard_log="app/data/logs",
        )

        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True
        )

        print("Training completed!")
        self.model.save("child_policy_model")
        print("Model saved as 'child_policy_model'")
        learning_env.close()

    def load_model(self, path="child_policy_model"):
        try:
            self.model = MaskablePPO.load(path)
            print(f"Model loaded from {path}")
        except FileNotFoundError:
            print(f"Model file not found at {path}. Using heuristic policy instead.")
            self.model = None
