from app.simulation.activity.ServerActivity import ServerActivity

class Break(ServerActivity):
    def __init__(self, start: float, stop: float, server_id: int, expected_stop: float, 
                 break_type: int, expected_start: float):
        super().__init__(start, stop, server_id, expected_stop)
        self.break_type = break_type
        self.expected_start = expected_start

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        
        return (
            self.break_type == other.break_type and
            self.expected_start == other.expected_start
        )