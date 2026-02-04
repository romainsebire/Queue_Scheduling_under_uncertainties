import heapq
from app.simulation.events.Event import Event
import itertools



class Events: 
    
    # To sort events with the same sim_time, whichever is added first, is first
    _counter = itertools.count()

    def __init__(self):
        self._heap = []

    def add(self, event: Event):
        """
        Add a new event in Events sorted by sim_time.

        Parameters:
            event(Event): Event to add to Events.
        """

        heapq.heappush(self._heap, (event.sim_time, next(Events._counter), event))

    def next_batch(self) -> list:
        """
        Return and delete next events.

        Returns:
            list of next event(Event): return the next events (same sim time).
                If Events empty, return None. 
        """
        if not self._heap:
            return []

        # sim time of the first event
        sim_time = self._heap[0][0]
        batch = []

        while self._heap and self._heap[0][0] == sim_time:
            batch.append(heapq.heappop(self._heap)[2])

        return batch
    
    def remove(self, event: Event) -> bool:
        """
        Remove the first event equal to `event` (using ==).

        Returns:
            bool: True if an event was removed, False otherwise.
        """
        for i, (_, _, e) in enumerate(self._heap):
            if e == event:
                del self._heap[i]
                heapq.heapify(self._heap)
                return True
        return False


    def is_empty(self) -> bool:
        """
        Checks if Events is empty.

        Returns:
            bool: True is empty, False if there are events.
        """
        return not self._heap
    

    def __len__(self) -> int:
        return len(self._heap)
    
    def __contains__(self, event: Event) -> bool:
        return any(e == event for _, _, e in self._heap)

