"""
Evaluation script — runs the trained policy on 50 instances and computes the final grade.

Usage:
    python evaluate.py                        # default: fourth solution
    python evaluate.py --solution fourth
    python evaluate.py --solution third
    python evaluate.py --solution second
    python evaluate.py --solution first
"""

import argparse
import importlib
import os

import gymnasium as gym
import pandas as pd
from gymnasium.envs.registration import register
from sb3_contrib import MaskablePPO

from app.data.Instance import Instance
from app.simulation.envs.Env import Env
from app.simulation.policies.PolicyEvaluation import PolicyEvaluation

# ── Solution registry (mirrors app/main.py) ────────────────────────────────────
SOLUTIONS = {
    "first": {
        "env_entry":     "app.simulation.envs.ChildEnvV1:ChildEnv",
        "policy_module": "app.simulation.policies.ChildPolicyV1",
        "model_path":    "app/data/models/ppo_first",
    },
    "second": {
        "env_entry":     "app.simulation.envs.ChildEnvV2:ChildEnv",
        "policy_module": "app.simulation.policies.ChildPolicyV2",
        "model_path":    "app/data/models/ppo_second",
    },
    "third": {
        "env_entry":     "app.simulation.envs.ChildEnvV3:ChildEnv",
        "policy_module": "app.simulation.policies.ChildPolicyV3",
        "model_path":    "app/data/models/ppo_third",
    },
    "fourth": {
        "env_entry":     "app.simulation.envs.ChildEnvV4:ChildEnv",
        "policy_module": "app.simulation.policies.ChildPolicyV4",
        "model_path":    "app/data/models/ppo_fourth",
    },
}

root_path    = "./instance_set"
instance_set = range(50)
NB_RUNS_INST = 1


def check_solution(instance, solution_path):
    is_valid, error = True, None
    df = pd.read_csv(solution_path, sep=";")
    eps = 1e-4

    if (df["start"] < df["arrival"]).any():
        invalid_start = df[df["start"] < df["arrival"]]
        return False, f"Some rows have start before arrival:\n{invalid_start}"

    if df["client"].duplicated().any():
        duplicates = df[df.duplicated(subset="client", keep=False)]
        return False, f"Some clients appear more than once:\n{duplicates}"

    if ((df["start"] + df["real_proc_time"] - df["end"]).abs() > eps).any():
        invalid_end = df[(df["start"] + df["real_proc_time"] - df["end"]).abs() > eps]
        return False, f"Some service have incoherent end:\n{invalid_end}"

    customers = Env._create_customers_from_steps(instance.timeline)
    servers   = Env._build_servers_from_average_matrix(instance.average_matrix)
    customer_ids = set(customers.keys())
    server_ids   = set(servers.keys())

    if not set(df["client"]).issubset(customer_ids):
        bad = set(df["client"]) - customer_ids
        return False, f"Invalid client ids: {bad}"

    if not set(df["server"]).issubset(server_ids):
        bad = set(df["server"]) - server_ids
        return False, f"Invalid server ids: {bad}"

    df = df.copy()
    df["expected_arrival"] = df["client"].map(
        {cid: c.arrival_time for cid, c in customers.items()}
    )
    if not (df["arrival"] == df["expected_arrival"]).all():
        bad = df[df["arrival"] != df["expected_arrival"]]
        return False, f"Invalid arrival time:\n{bad}"

    df["expected_service"] = df.apply(
        lambda r: customers[r.client].real_service_times[r.server], axis=1
    )
    if ((df["expected_service"] - df["real_proc_time"]).abs() > eps).any():
        bad = df[(df["expected_service"] - df["real_proc_time"]).abs() > eps]
        return False, f"Invalid service duration:\n{bad}"

    df = df.sort_values(["server", "start"])
    overlap = df["start"] < df.groupby("server")["end"].shift()
    if overlap.any():
        bad = df[overlap]
        return False, f"Server overlap detected:\n{bad[['server', 'start', 'end']]}"

    return is_valid, error


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL policy on 50 instances")
    parser.add_argument(
        "--solution",
        choices=list(SOLUTIONS.keys()),
        default="fourth",
        help="Which solution version to evaluate (default: fourth)",
    )
    args = parser.parse_args()

    cfg = SOLUTIONS[args.solution]
    print(f"[evaluate] solution={args.solution} | model={cfg['model_path']}.zip")

    # ── Load model ─────────────────────────────────────────────────────────────
    model_zip = cfg["model_path"] + ".zip"
    if not os.path.exists(model_zip):
        print(f"ERROR: Trained model not found at {model_zip}")
        print(f"  Run first: python -m app.main --solution {args.solution} --mode train")
        exit(1)

    policy_module = importlib.import_module(cfg["policy_module"])
    model = policy_module.ChildPolicy(f"Evaluate-{args.solution}")
    model.model = MaskablePPO.load(cfg["model_path"])
    print(f"Model loaded successfully from {model_zip}")

    # ── Register env ───────────────────────────────────────────────────────────
    register(id="ChildEnv", entry_point=cfg["env_entry"])

    # ── Evaluation loop ────────────────────────────────────────────────────────
    os.makedirs("./results/tmp", exist_ok=True)

    score    = 0
    is_valid = True

    for instance_id in instance_set:
        instance = Instance.create(
            Instance.SourceType.FILE,
            f"{root_path}/timeline_{instance_id}.json",
            f"{root_path}/average_matrix_{instance_id}.json",
            f"{root_path}/appointments_{instance_id}.json",
            f"{root_path}/unavailability_{instance_id}.json",
        )
        for run in range(NB_RUNS_INST):
            print(f"\nEvaluation of instance {instance_id}, run {run}")
            env      = gym.make("ChildEnv", mode=Env.MODE.TEST, instance=instance)
            sol_file = f"result_tmp_{instance_id}_{run}.csv"

            model.simulate(env, print_logs=False, save_to_csv=True,
                           path="./results/tmp/", file_name=sol_file)

            policy_evaluation = PolicyEvaluation(
                instance.timeline,
                instance.appointments,
                clients_history=model.customers_history,
            )
            policy_evaluation.evaluate()

            is_valid, error = check_solution(instance, "./results/tmp/" + sol_file)
            if not is_valid:
                score = -1
                print(f"Error in instance {instance_id}: {error}")
                break

            score += policy_evaluation.final_grade

        if not is_valid:
            break

    if is_valid:
        score /= len(instance_set) * NB_RUNS_INST

    print(f"\nFinal score: {score}")


if __name__ == "__main__":
    main()
