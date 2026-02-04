from app.simulation.events.Event import Event

class ServerOpeningEvent(Event):
    def __init__(self, sim_time: float, server_id: int):
        super().__init__(sim_time)
        self.server_id = server_id

    def __eq__(self, event):
        if not super().__eq__(event):
            return False
        
        if not self.server_id == event.server_id:
            return False
        
        return True
    
    def apply(self, env):
        """
        Server opening event means that an activity ended. 
        The activity must be removed.
        """
        env.terminate_activity(self.server_id)
        
