from app.simulation.events.Event import Event

class PendingActivityEvent(Event):
    def __init__(self, sim_time: float, activity_id: int):
        super().__init__(sim_time)
        self.activity_id = activity_id

    def __eq__(self, event):
        if not super().__eq__(event):
            return False
        
        if not self.activity_id == event.activity_id:
            return False
        
        return True
    
    def apply(self, env):
        """
        A server is closing due to a pending activity that needs to be moved to current activity.
        """
        env.handle_pending_activity(self.activity_id)