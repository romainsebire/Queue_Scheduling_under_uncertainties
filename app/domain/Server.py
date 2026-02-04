class Server:
    def __init__(self, id: int, avg_service_time: dict):
        self.id = id
        self.avg_service_time = avg_service_time
        self.working_time = 0

    def __eq__(self, other):
        if not isinstance(other, Server):
            return NotImplemented
        
        return (
            self.id == other.id and
            self.avg_service_time == other.avg_service_time and
            self.working_time == other.working_time
        )
    
    def increase_working_time(self, duration: float):
        self.working_time += duration