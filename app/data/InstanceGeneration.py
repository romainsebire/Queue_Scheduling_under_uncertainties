import numpy as np
import json
import os
from app.data.breaks.Breaks import Breaks
from app.data.Scenario import Scenario

class InstanceGeneration:
    def __init__(self, scenario: Scenario):
        """
        Class initialization with all required parameters.

        Parameters:
            config(ScenarioConfig): configuration with all required parameters
        """
        # Initializes parameters
        self.S = scenario.S
        self.C = scenario.C
        self.lmbd = scenario.lmbd
        self.avg_low = scenario.avg_low
        self.avg_high = scenario.avg_high
        self.deviation_low = scenario.deviation_low
        self.deviation_high = scenario.deviation_high
        self.unbearable_wait = scenario.unbearable_wait 
        self.unbearable_wait_appointment = scenario.unbearable_wait_appointment
        self.max_arrival_time = scenario.max_arrival_time
        self.max_sim_time = scenario.max_sim_time
        assert scenario.p_appointment>=0 and scenario.p_appointment<=1, "p_appointment must be between 0 and 1."
        self.p_appointment = scenario.p_appointment
        self.dev_appointment = 10
        assert scenario.p_no_show>=0 and scenario.p_no_show<=1, "p_no_show must be between 0 and 1."
        self.p_no_show = scenario.p_no_show
        assert scenario.p_unavailability>=0 and scenario.p_unavailability<=1, "p_unavailability must be between 0 and 1."
        self.p_unavailability = scenario.p_unavailability
        self.mean_time_between_arrivals = scenario.mean_time_between_arrivals

        # Gives the seed to a numpy independant generator
        self.rng = np.random.default_rng(scenario.seed)


    def _gen_average_matrix(self) -> None:
        """
        Generate S x C matrix of averages (mean values) between services (S) and clients types (C).
        """
        # Initialize an empty matrix
        self.average_matrix = [[0]*self.C for _ in range(self.S)]
        for i in range(self.S):
            for j in range(self.C):
                # Random integer value in [avg_low, avg_high)
                self.average_matrix[i][j] = self.rng.integers(low=self.avg_low, high=self.avg_high)  
    
    def _gen_deviation_matrix(self) -> None:
        """
        Generate an S x C matrix of standard deviations (deviation values) corresponding to the averages in the average matrix.
        """
        # Initialize an empty matrix
        self.deviation_matrix = [[0]*self.C for _ in range(self.S)]  
        for i in range(self.S):
            for j in range(self.C):
                # Random integer value in [deviation_low, deviation_high)
                self.deviation_matrix[i][j] = self.rng.integers(low=self.deviation_low, high=self.deviation_high)  
        
    def _gen_file(self) -> None:
        """
        Generates client data for T steps, following a Poisson distribution for arrivals 
        and normally distributed estimated processing times
        """
        steps = []  # List of time steps
        appointments = []
        client_id = 0  # Unique client identifier 
        # first arrival time generation
        t = self.rng.poisson(self.mean_time_between_arrivals)
        while t < self.max_arrival_time:   ## TODO: replace by a law which determines the next arrival randomly, not at a full minute
            nb_client = self.rng.poisson(self.lmbd)  # Number of clients arriving at this step (Poisson)
            for _ in range(nb_client):
                client_need = self.rng.integers(0, self.C)  # Random client need (client index)
                # Determines if the customer comes with an appointment
                is_appointment = self.rng.binomial(n=1, p=self.p_appointment)
                if is_appointment:
                    # Draw an appointment time from the arrival time
                    # And make sure it is after 0
                    appointment_time = max(0, self.rng.normal(t, self.dev_appointment))
                # For appointments
                if is_appointment == 1:
                    # Add the appointments
                    appointments.append([client_id, client_need, appointment_time])
                    
                    current_unbearable_wait = self.unbearable_wait_appointment

                    # Deal with no shows
                    # If no shows, don't save the customer in arrivals
                    is_no_show = self.rng.binomial(n=1, p=self.p_no_show)
                    if is_no_show == 1:
                        client_id += 1
                        continue
                else:
                    current_unbearable_wait = self.unbearable_wait

                estimated_process_time_row = [0]*self.S  # Initialize estimated times for each service
                for i in range(self.S):
                    # Draw a processing time from a normal distribution centered on the mean with the corresponding std deviation
                    estimated_time = self.rng.normal(
                        self.average_matrix[i][client_need],
                        self.deviation_matrix[i][client_need]
                    )
                    # Round and ensure the time is at least 1
                    estimated_process_time_row[i] = max(1, round(estimated_time)) # TOCHECK: need to round here ?
                # Calculate abandonment time
                if current_unbearable_wait > 0: # Abandonment simulated
                    abandonment_time = self.rng.exponential(
                        current_unbearable_wait, # average: current time + unbearable wait 
                    )
                else:
                    abandonment_time = 0

                # Add to the step: [client ID, client need, list of estimated times, abandonment_time]
                steps.append([client_id, client_need, estimated_process_time_row, t, t+abandonment_time])

                client_id += 1  # Increment client ID
            # next arrivals time
            t = t + self.rng.poisson(self.mean_time_between_arrivals) 
        self.gen_file_data = steps
        self.appointments = appointments

    def _gen_break_duration(self, avg_break_duration: int) -> float:
        """
        Generate the break duration.
        
        Parameters:
            avg_break_duration (int): Average duration of the break.
        
        Returns: 
            (float) break duration, if 0: no break
        """

        break_duration = self.rng.normal(
            avg_break_duration, 
            avg_break_duration /3
        )
        return max(0, break_duration)


    def _gen_unavailability(self):
        """
        Generate matrices determining unavailabilities for servers.
        
        """
        unavailability = []
        breaks = Breaks()
        for _ in range(self.S):
            current_unavailability = []
            previous_break_end = 0

            # Delays: small break
            # Determines if there is a break or not
            if self.rng.binomial(n=1, p=self.p_unavailability) == 1:
                delay = self._gen_break_duration(breaks.get_durations(Breaks.BREAK_ID.SMALL))
                if delay > 0:
                    current_unavailability.append([0, delay, Breaks.BREAK_ID.SMALL])
                    previous_break_end = delay

            # Early departure
            if self.rng.binomial(n=1, p=self.p_unavailability) == 1:
                delay = self._gen_break_duration(breaks.get_durations(Breaks.BREAK_ID.SMALL))
                if delay > 0:
                    current_unavailability.append([self.max_sim_time-delay, 
                                                   self.max_sim_time, 
                                                   Breaks.BREAK_ID.SMALL])
            
            # General breaks
            
            if self.max_sim_time == 630:
                mean_times = [120, 270, 480]
                mean_durations = [breaks.get_durations(Breaks.BREAK_ID.SMALL), 
                                  breaks.get_durations(Breaks.BREAK_ID.LONG), 
                                  breaks.get_durations(Breaks.BREAK_ID.SMALL)]
                break_type = [Breaks.BREAK_ID.SMALL, Breaks.BREAK_ID.LONG, Breaks.BREAK_ID.SMALL]
            else:
                mean_times = [self.max_sim_time/2]
                mean_durations = [breaks.get_durations(Breaks.BREAK_ID.SMALL)]
                break_type = [Breaks.BREAK_ID.SMALL]

            for i in range(len(mean_durations)):
                 if self.rng.binomial(n=1, p=self.p_unavailability) == 1:
                    
                    relative_break_start = self.rng.exponential(max(0, mean_times[i]-previous_break_end))
                    delay = self._gen_break_duration(mean_durations[i])
                    if delay > 0:
                        break_end = previous_break_end+relative_break_start+delay
                        current_unavailability.append([previous_break_end+relative_break_start, break_end, break_type[i]])
                        previous_break_end = break_end
                        

            unavailability.append(current_unavailability)
        self.unavailability = unavailability

    def generate_instance(self):
        """
        Generate instance: timeline and average service time matrices
        """
        # Generation of mean and standard deviation matrices
        self._gen_average_matrix()
        self._gen_deviation_matrix()

        # Generation of client data from the previously created matrices
        self._gen_file()

        # Generation of client unavailability
        self._gen_unavailability()

        # Conversion of data into native Python types for JSON saving
        self.gen_file_data_native = convert_to_native(self.gen_file_data)
        self.matrices_data = convert_to_native(self.average_matrix)
        self.appointments_native = convert_to_native(self.appointments) 
        self.unavailability_native = convert_to_native(self.unavailability)

    def generate_files(self, output_dir = "app/data/data_files", extension = "") -> None:
        """
        Generate two files: timeline{extension}.json and average_matrix{extension}.json
        in the directory output_dir.

        Parameters:
            output_dir (str): Directory where to save the files.
                              Default: "data_files"
            extension (str): Extension in the name of the files.
                             Default: ""
            
        """
        # Creates the output dir if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        self.generate_instance()

        # Saving the generated data into a JSON file in the 'data_files' folder
        with open(os.path.join(output_dir, f"timeline{extension}.json"), "w") as f:
            json.dump(self.gen_file_data_native, f, indent=2)
        with open(os.path.join(output_dir, f"average_matrix{extension}.json"), "w") as f_mat:
            json.dump(self.matrices_data, f_mat, indent=2)
        with open(os.path.join(output_dir, f"appointments{extension}.json"), "w") as f_mat:
            json.dump(self.appointments_native, f_mat, indent=2)
        with open(os.path.join(output_dir, f"unavailability{extension}.json"), "w") as f_mat:
            json.dump(self.unavailability_native, f_mat, indent=2)

        # Message indicating that the files have been saved
        print(f"Files saved in the folder '{output_dir}/'")


            
def convert_to_native(obj):
    """
    Recursively converts numpy objects to native Python types for JSON serialization

    Parameters:
        obj (list or np.floating or np.interger or other): The object to convert. Can be a nested list containing numpy floating types.
    
    Returns:
        Converted object with all numpy floating and interger types replaced by native Python floats.
    """
    if isinstance(obj, list):
        return [convert_to_native(x) for x in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):  
        return int(obj)
    else:
        return obj