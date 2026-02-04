class Customer:
    def __init__(self, id: int, arrival_time: float, task: int, 
                 real_service_times: dict, abandonment_time: float):
        
        assert arrival_time <= abandonment_time, "Abandonment time must be greater or equal to arrival time."

        self.id = id
        self.arrival_time = arrival_time
        self.task = task
        self.real_service_times = real_service_times
        self.abandonment_time = abandonment_time

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return (
            self.id == other.id and
            self.arrival_time == other.arrival_time and
            self.task == other.task and
            self.real_service_times == other.real_service_times and
            self.abandonment_time == other.abandonment_time
        )