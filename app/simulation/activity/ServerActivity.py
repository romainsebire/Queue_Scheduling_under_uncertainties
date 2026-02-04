from abc import ABC
import itertools

class ServerActivity(ABC):
    _id_gen = itertools.count()

    def __init__(self, start: float, stop: float, server_id: int, expected_stop: float):
        self.id = next(ServerActivity._id_gen)
        self.start = start
        self.stop = stop
        self.server_id = server_id
        self.expected_stop = expected_stop
        
    def get_duration(self):
        return self.stop - self.start
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return (
            self.start == other.start and
            self.stop == other.stop and
            self.server_id == other.server_id and
            self.expected_stop == other.expected_stop
        )
    
