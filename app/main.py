from pyexpat import model
from app.simulation.policies.ChildPolicy import ChildPolicy
from app.data.Instance import Instance
import gymnasium as gym
from app.simulation.policies.PolicyEvaluation import PolicyEvaluation
from gymnasium.envs.registration import register
from app.data.Scenario import Scenario
from app.simulation.envs.Env import Env
from app.simulation.envs.ChildEnv import ChildEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# ----- Save in Gym -----
register(
    id="Child_Env",
    entry_point="app.simulation.envs.ChildEnv:ChildEnv", 
)

def main():
    scenario = Scenario.from_json("app/data/config/queue_config.json")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,                  # Fréquence de sauvegarde
        save_path='./logs/checkpoints/',   # Dossier de destination
        name_prefix='ppo_model'            # Préfixe du nom de fichier
    )

    model = ChildPolicy("Env")

    model.learn(scenario,250_000, callback=checkpoint_callback, verbose=1)

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