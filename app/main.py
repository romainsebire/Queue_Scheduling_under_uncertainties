from app.simulation.policies.ChildPolicy import ChildPolicy
from app.data.Instance import Instance
import gymnasium as gym
from app.simulation.policies.PolicyEvaluation import PolicyEvaluation
from gymnasium.envs.registration import register
from app.data.Scenario import Scenario
from app.simulation.envs.Env import Env
#from app.simulation.envs.ChildEnv import ChildEnv

# ----- Save in Gym -----
register(
    id="Child_Env",
    entry_point="app.simulation.envs.ChildEnv:ChildEnv", 
)

def main():
    scenario = Scenario.from_json("app/data/config/queue_config.json")
    
    model = ChildPolicy("Env")

    model.learn(scenario, 10000, 1)

    instance = Instance.create(Instance.SourceType.FILE,
                    "app/data/data_files/timeline_0.json", 
                    "app/data/data_files/average_matrix_0.json",
                    "app/data/data_files/appointments_0.json", 
                    "app/data/data_files/unavailability_0.json")
    env = gym.make("Child_Env", mode=Env.MODE.TEST, instance=instance)
    model.simulate(env, print_logs=True, save_to_csv=True, path="app/data/results/", file_name="result_0.csv")
    policy_evaluation = PolicyEvaluation(instance.timeline, instance.appointments, clients_history=model.customers_history)
    policy_evaluation.evaluate()

if __name__ == "__main__":
    main()