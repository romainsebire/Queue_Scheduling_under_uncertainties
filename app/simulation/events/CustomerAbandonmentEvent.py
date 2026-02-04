from app.simulation.events.Event import Event

class CustomerAbandonmentEvent(Event):
    def __init__(self, sim_time: float, customer_id: int):
        super().__init__(sim_time)
        self.customer_id = customer_id

    def __eq__(self, event):
        if not super().__eq__(event):
            return False
        
        if not self.customer_id == event.customer_id:
            return False
        
        return True
    
    def apply(self, env):
        """
        Customer abandonment Event is made to remove a customer to waiting customers.
        """
        env.remove_waiting_customer(self.customer_id)
        env.increase_customer_abandonment_count()