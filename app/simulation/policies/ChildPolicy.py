import os
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from app.simulation.policies.Policy import Policy
from app.simulation.envs.Env import Env

class ChildPolicy(Policy):
    def __init__(self, model_title):
        super().__init__(model_title)
        self.model = None
        # Path where the trained model will be saved
        self.model_path = os.path.join("app", "data", "models", "ppo_child_policy")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def _mask_fn(self, env):
        """
        Helper function to extract the action mask from the environment.
        Required for the ActionMasker wrapper.
        """
        return env.unwrapped.action_masks()

    def learn(self, scenario, total_timesteps, verbose):
        """
        Trains the RL agent using MaskablePPO.
        """
        print(f"--- Starting Training for {total_timesteps} steps ---")
        
        # 1. Create the training environment
        env = gym.make("Child_Env", mode=Env.MODE.TRAIN, scenario=scenario)
        
        # 2. ActionMasker wrapper (CRUCIAL)
        # This allows the algorithm to see which actions are forbidden at each step
        env = ActionMasker(env, self._mask_fn)

        # 3. Model initialization
        # "MultiInputPolicy" is required because the observation is a Dict (queue, servers, context)
        self.model = MaskablePPO(
            "MultiInputPolicy", 
            env, 
            verbose=verbose,
            learning_rate=3e-4,
            gamma=0.99,            # Discount factor (future reward weighting)
            ent_coef=0.01,         # Encourages some exploration at the beginning
            batch_size=64
        )

        # 4. Start training
        self.model.learn(total_timesteps=total_timesteps)
        
        # 5. Save the model
        self.model.save(self.model_path)
        print(f"--- Training Finished. Model saved at {self.model_path}.zip ---")

    def _predict(self, obs, info):
        """
        Used during the simulation/test phase (main.py -> simulate).
        """
        # 1. Load the model if not already in memory
        if self.model is None:
            if os.path.exists(self.model_path + ".zip"):
                # Reload the model without the environment for inference
                self.model = MaskablePPO.load(self.model_path)
            else:
                # Fallback: If no model is found (e.g., testing without training), choose a valid random action
                # This is a safety mechanism.
                mask = info.get("action_mask")
                valid_actions = [i for i, valid in enumerate(mask) if valid]
                return np.random.choice(valid_actions)

        # 2. Retrieve the action mask from info
        # In Env._get_info(), you return "action_mask", which is perfect.
        action_masks = np.array(info.get("action_mask"), dtype=bool)

        # 3. Prediction
        # deterministic=True avoids randomness during testing (we want the best learned action)
        action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
        
        return int(action)