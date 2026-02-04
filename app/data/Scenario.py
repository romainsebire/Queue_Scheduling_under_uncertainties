import json
import itertools

class Scenario:

    KEY_MAP = {
        "num_servers": "S",
        "C": "C",
        "lambdas": "lmbd",
        "avg_low": "avg_low",
        "avg_high": "avg_high",
        "deviation_low": "deviation_low",
        "deviation_high": "deviation_high",
        "unbearable_wait": "unbearable_wait",
        "max_arrival_time": "max_arrival_time",
        "max_sim_time": "max_sim_time",
        "p_appointment": "p_appointment",
        "p_no_show": "p_no_show",
        "p_unavailability": "p_unavailability",
        "mean_time_between_arrivals": "mean_time_between_arrivals",
        "unbearable_wait_appointment": "unbearable_wait_appointment",
    }

    def __init__(self, S: int, C: int, 
                 lmbd: float, avg_low: int, avg_high: int, 
                 deviation_low: int, deviation_high: int, 
                 unbearable_wait: int = 0, 
                 max_arrival_time: int = 600, max_sim_time: int = 630,
                 p_appointment:float = 0, p_no_show:float = 0,
                 p_unavailability = 0, mean_time_between_arrivals = 1,
                 unbearable_wait_appointment: int = 0,
                 seed = None):
        """
        Class initialization with all required parameters and required checks.

        Parameters:
            S (int): Total number of servers 
                    (e.g., 30).
            C (int): Total number of client types or customer categories 
                    (e.g., 50).
            lmbd (float): Lambda parameter of the Poisson distribution, 
                        representing the average number of clients 
                        arriving at each time step (e.g., 0.5).
            avg_low (int): Minimum value for generating mean service times 
                        between servers and client types (e.g., 5).
            avg_high (int): Maximum value for generating mean service times 
                            between servers and client types (e.g., 27).
            deviation_low (int): Minimum value for generating standard 
                                deviations of service times (e.g., 1).
            deviation_high (int): Maximum value for generating standard 
                                deviations of service times (e.g., 3).
            unbearable_wait (int): Unbearable wait which defines average abandonment time (exp distribution).
                                Default 0: No abandonment. 
            max_arrival_time (int): Maximum time for customers arrivals.
                    Default 600.
            max_sim_time (int): Maximum simulation time.
                    Default 630.
            p_appointment (float): Probability for the customer to arrive with an appointment.
                                Default 0: No appointments. 
            p_no_show (float): Probability for the customer with an appointment to never arrive. 
                                Default 0: Every appointment arrives. 
            p_unavailability (float): Probability for the server to be unavailable.
                                Default 0: Servers always available. 
            mean_time_between_arrivals (float): Average time between two batch of arrivals.
                                Default 1: 1 minute
            unbearable_wait_appointment (int): Unbearable wait which defines average abandonment time (exp distribution) for appointments.
                                Default 0: No abandonment. 
            seed (int, optional): Random seed for reproducibility. 
                                Default is None.
        """
        # Initializes parameters
        self.S = S
        self.C = C
        self.lmbd = lmbd
        self.avg_low = avg_low
        self.avg_high = avg_high
        self.deviation_low = deviation_low
        self.deviation_high = deviation_high
        self.unbearable_wait = unbearable_wait 
        self.unbearable_wait_appointment = unbearable_wait_appointment
        assert max_arrival_time <= max_sim_time, "Customer cannot arrive after the end of the simulation."
        self.max_arrival_time = max_arrival_time
        self.max_sim_time = max_sim_time
        assert p_appointment>=0 and p_appointment<=1, "p_appointment must be between 0 and 1."
        self.p_appointment = p_appointment
        self.dev_appointment = 10
        assert p_no_show>=0 and p_no_show<=1, "p_no_show must be between 0 and 1."
        self.p_no_show = p_no_show
        assert p_unavailability>=0 and p_unavailability<=1, "p_unavailability must be between 0 and 1."
        self.p_unavailability = p_unavailability
        self.mean_time_between_arrivals = mean_time_between_arrivals

        self.seed = seed

    @classmethod
    def from_json(cls, json_path: str, seed: int | None = None):
        scenario = Scenario.from_json_many(json_path, seed)
        assert len(scenario) == 1, "There should be only one scenario."

        return scenario[0]

    @classmethod
    def from_json_many(cls, json_path: str, seed: int | None = None):
        with open(json_path, "r") as f:
            cfg = json.load(f)

        # Convert scalars → list
        expanded_cfg = {}
        for json_key, cls_key in Scenario.KEY_MAP.items():
            if json_key in cfg:
                value = cfg[json_key]
                expanded_cfg[cls_key] = (
                    value if isinstance(value, list) else [value]
                )

        scenarios = []

        # Produit cartésien des paramètres
        keys = expanded_cfg.keys()
        values_product = itertools.product(*expanded_cfg.values())

        for values in values_product:
            kwargs = dict(zip(keys, values))
            kwargs["seed"] = seed
            scenarios.append(cls(**kwargs))

        return scenarios