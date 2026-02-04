from abc import ABC, abstractmethod

class Event(ABC):
    def __init__(self, sim_time: float):
        self.sim_time = sim_time

    @abstractmethod
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.sim_time == other.sim_time
    
    @abstractmethod
    def apply(self, env) -> None:
        raise NotImplementedError