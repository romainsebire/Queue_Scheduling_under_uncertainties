"""
Queue Scheduling RL — main entry point

Usage:
    python -m app.main                                       # fourth solution, train mode
    python -m app.main --solution fourth --mode train
    python -m app.main --solution fourth --mode evaluate
    python -m app.main --solution third  --mode train
    python -m app.main --solution second --mode train
    python -m app.main --solution first  --mode train
    python -m app.main --solution fourth --mode train --seed 42 --timesteps 250000

Solutions (action spaces):
    first  — Patient ID         (~60%  score)
    second — Task ID            (~70%  score)
    third  — VIP/Walk-in + ROI  (~90%  score)
    fourth — 4 Semantic actions (~92.5% score)  [default]
"""

import argparse
import importlib
import os
import random

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

from app.data.Instance import Instance
from app.data.Scenario import Scenario
from app.simulation.envs.Env import Env
from app.simulation.policies.PolicyEvaluation import PolicyEvaluation

# ── Solution registry ──────────────────────────────────────────────────────────
SOLUTIONS = {
    "first": {
        "env_entry":     "app.simulation.envs.ChildEnvV1:ChildEnv",
        "policy_module": "app.simulation.policies.ChildPolicyV1",
    },
    "second": {
        "env_entry":     "app.simulation.envs.ChildEnvV2:ChildEnv",
        "policy_module": "app.simulation.policies.ChildPolicyV2",
    },
    "third": {
        "env_entry":     "app.simulation.envs.ChildEnvV3:ChildEnv",
        "policy_module": "app.simulation.policies.ChildPolicyV3",
    },
    "fourth": {
        "env_entry":     "app.simulation.envs.ChildEnvV4:ChildEnv",
        "policy_module": "app.simulation.policies.ChildPolicyV4",
    },
}

DEFAULT_SEED      = 42
DEFAULT_TIMESTEPS = 250_000
DEFAULT_SOLUTION  = "fourth"


def set_seeds(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="RL Queue Scheduling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--solution",
        choices=list(SOLUTIONS.keys()),
        default=DEFAULT_SOLUTION,
        help=f"Action space version to use (default: {DEFAULT_SOLUTION})",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate"],
        default="train",
        help="train = train then evaluate | evaluate = load saved model and evaluate (default: train)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"Training timesteps (default: {DEFAULT_TIMESTEPS})",
    )
    args = parser.parse_args()

    # ── Reproducibility ────────────────────────────────────────────────────────
    set_seeds(args.seed)
    print(f"[config] solution={args.solution} | mode={args.mode} | "
          f"seed={args.seed} | timesteps={args.timesteps}")

    # ── Dynamic env registration ───────────────────────────────────────────────
    cfg = SOLUTIONS[args.solution]
    register(id="ChildEnv", entry_point=cfg["env_entry"])

    # ── Dynamic policy import ──────────────────────────────────────────────────
    policy_module = importlib.import_module(cfg["policy_module"])
    ChildPolicy = policy_module.ChildPolicy

    scenario = Scenario.from_json("app/data/config/queue_config.json")
    model = ChildPolicy(f"Solution-{args.solution}", seed=args.seed)

    # ── Train ──────────────────────────────────────────────────────────────────
    if args.mode == "train":
        model.learn(scenario, total_timesteps=args.timesteps, verbose=1)
    else:
        # evaluate-only: warn if no saved model exists
        model_zip = model.model_path + ".zip"
        if not os.path.exists(model_zip):
            print(f"[WARNING] No trained model found at {model_zip}")
            print(f"  Run first: python -m app.main --solution {args.solution} --mode train")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    instance = Instance.create(
        Instance.SourceType.FILE,
        "app/data/data_files/timeline_0.json",
        "app/data/data_files/average_matrix_0.json",
        "app/data/data_files/appointments_0.json",
        "app/data/data_files/unavailability_0.json",
    )
    env = gym.make("ChildEnv", mode=Env.MODE.TEST, instance=instance)
    model.simulate(env, print_logs=True, save_to_csv=True,
                   path="app/data/results/", file_name="result_0.csv")

    policy_evaluation = PolicyEvaluation(
        instance.timeline,
        instance.appointments,
        clients_history=model.customers_history,
    )
    policy_evaluation.evaluate()


if __name__ == "__main__":
    main()
