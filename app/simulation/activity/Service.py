from app.simulation.activity.ServerActivity import ServerActivity

class Service(ServerActivity):
    def __init__(self, start: float, stop: float, server_id: int, expected_stop: float, 
                 customer_id: int, task_id: int):
        super().__init__(start, stop, server_id, expected_stop)
        self.customer_id = customer_id
        self.task_id = task_id

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        
        return (
            self.customer_id == other.customer_id and
            self.task_id == other.task_id
        )
