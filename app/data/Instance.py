from enum import Enum, auto
from app.data.Scenario import Scenario
from app.data.InstanceGeneration import InstanceGeneration
from app.utils import extract_data

class Instance:

    class SourceType(Enum):
        FILE = auto()
        CONFIG = auto()

    def __init__(self, timeline: list, average_matrix: list, 
                 appointment: list, unavailability: list,
                 time_limit: int = 630):
        
        # both average_matrix and unavailability have servers as first dimension
        # they should have the same len
        assert len(average_matrix) == len(unavailability), "average_matrix and unavailability should have the same size."

        # clients arrival timeline
        self.timeline = timeline
        # average processing time for the classes by the server
        self.average_matrix = average_matrix
        # appointments
        self.appointments = appointment
        # servers unavailability
        self.unavailability = unavailability

        # number of servers according to Kendall's notation
        self.C = len(self.average_matrix)
        # number of possible needs
        self.num_needs = len(self.average_matrix[0])
        # number of time steps
        if len(timeline) > 0:
            self.max_arrival_time = max(customer[3] for customer in timeline)
        else: 
            self.max_arrival_time = 0
        # time limit
        self.time_limit = time_limit

    @classmethod
    def create(cls, source_type: SourceType, 
                timeline_path: str = None, average_matrix_path: str = None, 
                appointment_path: str = None, unavailability_path: str = None, 
                scenario: Scenario = None, time_limit: int = 630):
        timeline = []
        average_matrix = []
        appointments = []
        unavailability = []
        if source_type == Instance.SourceType.FILE:
            assert timeline_path is not None, "To create a scenario from file, a timeline path is needed."
            assert average_matrix_path is not None, "To create a scenario from file, an average matrix path is needed."
            assert appointment_path is not None, "To create a scenario from file, an appointment path is needed."
            assert unavailability_path is not None, "To create a scenario from file, an unavailability path is needed."

            # Data extraction
            timeline = extract_data(timeline_path)
            average_matrix = extract_data(average_matrix_path)
            appointments = extract_data(appointment_path)
            unavailability = extract_data(unavailability_path)
        elif source_type == Instance.SourceType.CONFIG:
            assert scenario is not None, "To create a scenario from config, a config is needed."

            # Scenario generation
            generator = InstanceGeneration(scenario)
            generator.generate_instance()

            # Data extraction
            timeline = generator.gen_file_data_native
            average_matrix = generator.matrices_data
            appointments = generator.appointments_native
            unavailability = generator.unavailability_native
            time_limit = generator.max_sim_time
        else:
            raise ValueError("SourceType not supported.")

        return Instance(timeline, average_matrix, appointments, unavailability, 
                        time_limit)
        