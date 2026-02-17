from app.data.Instance import Instance
from app.simulation.envs.Env import Env
import gymnasium as gym
from app.simulation.policies.PolicyEvaluation import PolicyEvaluation
from app.simulation.envs.RandomEnv import RandomEnv
from app.simulation.policies.Random import Random
import pandas as pd
from app.domain.Server import Server
from gymnasium.envs.registration import register
from app.simulation.envs.ChildEnv import ChildEnv
from app.simulation.policies.ChildPolicy import ChildPolicy
from sb3_contrib import MaskablePPO



root_path = "./instance_set"
instance_set = range(50)
 
score = 0
NB_RUNS_INST = 1
is_valid = True

model = ChildPolicy("Model")
try:
    trained_model = MaskablePPO.load("./logs/checkpoints/ppo_model_70000_steps.zip")
    model.model = trained_model
    print("Modèle chargé avec succès.")
except FileNotFoundError:
    print("ERREUR : Le fichier '.zip' est introuvable.")
    exit()

register(
    id="Child_Env",
    entry_point="app.simulation.envs.ChildEnv:ChildEnv", 
)
env_id = "Child_Env"


def check_solution(instance, solution_path):
    is_valid, error = True, None
    df = pd.read_csv(solution_path, sep=";")
    eps = 1e-4

    # Check if the customer is served after arrival
    if (df["start"] < df["arrival"]).any():
            invalid_start = df[df["start"] < df["arrival"]]
            return False, f"Some rows have start before arrival:\n{invalid_start}"

    # Check if customer served more than once
    if df["client"].duplicated().any():
        duplicates = df[df.duplicated(subset="client", keep=False)]
        return False, f"Some clients appear more than once:\n{duplicates}"
    
    # Check the end is start + real_proc_time
    if ((df["start"] + df["real_proc_time"] - df["end"]).abs() > eps).any():
        invalid_end = df[
            (df["start"] + df["real_proc_time"] - df["end"]).abs() > eps
        ]
        return False, f"Some service have incoherent end:\n{invalid_end}"
        
    # Server and customer creation from insatnce
    customers = Env._create_customers_from_steps(instance.timeline)
    servers = Env._build_servers_from_average_matrix(instance.average_matrix)
    customer_ids = set(customers.keys())
    server_ids = set(servers.keys())

    # Invalid customer id
    if not set(df["client"]).issubset(customer_ids):
        bad = set(df["client"]) - customer_ids
        return False, f"Invalid client ids: {bad}"

    # Invalid server id
    if not set(df["server"]).issubset(server_ids):
        bad = set(df["server"]) - server_ids
        return False, f"Invalid server ids: {bad}"
	
    df = df.copy()

    # Check arrival time
    df["expected_arrival"] = df["client"].map(
        {cid: c.arrival_time for cid, c in customers.items()}
    )

    if not (df["arrival"] == df["expected_arrival"]).all():
        bad = df[df["arrival"] != df["expected_arrival"]]
        return False, f"Invalid arrival time:\n{bad}"

    # Check real service duration
    df["expected_service"] = df.apply(
        lambda r: customers[r.client].real_service_times[r.server],
        axis=1
    )

    if ((df["expected_service"] - df["real_proc_time"]).abs() > eps).any():
        bad = df[
            (df["expected_service"] - df["real_proc_time"]).abs() > eps
        ]
        return False, f"Invalid service duration:\n{bad}"
        
    # Check servers serves only one customer at once
    df = df.sort_values(["server", "start"])
    overlap = df["start"] < df.groupby("server")["end"].shift()

    if overlap.any():
        bad = df[overlap]
        return False, f"Server overlap detected:\n{bad[['server', 'start', 'end']]}"
    return is_valid, error
 
for instance_id in instance_set: 
	instance = Instance.create(Instance.SourceType.FILE,
				f"{root_path}/timeline_{instance_id}.json",
				f"{root_path}/average_matrix_{instance_id}.json",
				f"{root_path}/appointments_{instance_id}.json",
				f"{root_path}/unavailability_{instance_id}.json")
	for run in range(NB_RUNS_INST):     
		print(f"\n\nEvaluation of instance {instance_id} and run {run}")   
		env = gym.make(env_id, mode=Env.MODE.TEST, instance=instance)
		sol_file = f"result_tmp_{instance_id}_{run}.csv"
		model.simulate(env, print_logs=False, save_to_csv=True, path="./results/tmp/", file_name=sol_file)
		policy_evaluation = PolicyEvaluation(instance.timeline, instance.appointments, clients_history=model.customers_history)
		policy_evaluation.evaluate()
		is_valid, error = check_solution(instance, "./results/tmp/" + sol_file)
		if not is_valid:
			score = -1
			print(f"Error in instance {instance}, {error}")
			break

		score += policy_evaluation.final_grade
 
if is_valid:	
	score /= len(instance_set) * NB_RUNS_INST
 
print(f"\nFinal score: {score}")		