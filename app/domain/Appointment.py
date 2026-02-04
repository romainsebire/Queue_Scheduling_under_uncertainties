class Appointment:
    def __init__(self, time: float, customer_id: int, task_id: int):
        self.time = time
        self.customer_id = customer_id
        self.task_id = task_id
        self.service_time = None

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        
        return(
            self.time == other.time and
            self.customer_id == other.customer_id and
            self.task_id == other.task_id and
            self.service_time == other.service_time
        )
    
    def set_service_time(self, service_time: float):
        self.service_time = service_time