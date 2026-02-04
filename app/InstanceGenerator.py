from app.data.InstanceGeneration import InstanceGeneration
from app.data.Scenario import Scenario

config = Scenario.from_json("app/data/config/queue_config.json")
scenarioGeneration = InstanceGeneration(config)
scenarioGeneration.generate_files(extension="_10")