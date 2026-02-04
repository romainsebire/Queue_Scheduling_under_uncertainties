from app.domain.Customer import Customer
from app.domain.Appointment import Appointment
from app.simulation.envs.Env import Env

class PolicyEvaluation:
    def __init__(self, timeline, appointments, clients_history, 
                 unbearable_wait = 60, unbearable_wait_appointment = 30):
        self.customers= Env._create_customers_from_steps(timeline)
        self.appointments = Env._get_appointments_from_list(appointments)
        self.clients_history = clients_history
        self.unbearable_wait = unbearable_wait
        self.unbearable_wait_appointment = unbearable_wait_appointment
        self.epsilon_appointment = 3 # 3 minutes for epsilon appointment
        self.appointments_max_early = 60 # the appointment should not be taken before 1 hour early
    
    def _compute_waiting_score_mean(self) -> float:
        """
        Compute mean waiting score over all served clients.

        Returns:
            float: final grade
        """

        if not self.clients_history:
            return 0.0

        scores = []

        for c in self.clients_history:
            # Appointments score is calculated by itself
            if c["client"] not in self.appointments:
                wait_time = c["start"] - c["arrival"]

                if wait_time > self.unbearable_wait:
                    score = 0.0
                else:
                    score = 100 * (1 - wait_time / self.unbearable_wait)

                scores.append(score)

        # We need to add a score of zero for unserved customers
        number_unserved_clients = len(self.customers) - len(self.clients_history)

        return sum(scores) / (len(scores) + number_unserved_clients)
    
    def _get_customer_sevice_time(self, id: int) -> float:
        for c in self.clients_history:
            if c["client"] == id:
                return c["start"]
        
        return -1
    
    def _calculate_appointment_compliance(self):
        """
        Give a grade for appointment compliance.
        """
        if len(self.appointments) == 0:
            return 100
        
        scores = []
        no_valid_appointments = True
        for customer_id, appointment in self.appointments.items():
            # If the customer nerver arrived, it is not taken into account
            if customer_id not in self.customers:
                continue
            
            no_valid_appointments = False

            
            service_time = self._get_customer_sevice_time(customer_id)
            appointment_time = appointment.time

            # If the appointment has not been served
            if service_time == -1:
                scores.append(0.0)
                continue

            # If service time is around appointment with an error espilon, full grade
            if abs(service_time - appointment_time) <= self.epsilon_appointment:
                scores.append(100.0)
            elif service_time < appointment_time - self.epsilon_appointment and service_time > appointment_time - self.appointments_max_early:
                scores.append(100*
                            (1+
                            (service_time-appointment_time+self.epsilon_appointment)
                            /(self.appointments_max_early-self.epsilon_appointment)))
            elif service_time > appointment_time + self.epsilon_appointment and service_time < appointment_time + self.unbearable_wait_appointment:
                scores.append(100/(self.unbearable_wait_appointment-self.epsilon_appointment)
                            * 
                            (appointment_time-service_time+self.unbearable_wait_appointment))
            else:
                scores.append(0.0)

        if no_valid_appointments: 
            return 100
        
        return sum(scores) / len(scores)


    def evaluate(self):
        """
        Evaluate the model and give a grade on a 100%.
        Note that it is impossible to have a 100% in most cases.
        """
        weights = {
            "waiting": 0.4,
            "appointment": 0.4,
            "unserved": 0.2
        }

        # Customer waiting times
        self.grade_wait = self._compute_waiting_score_mean()

        # Appointment compliance
        self.grade_appointment = self._calculate_appointment_compliance()

        # Number of unserved customer
        self.grade_number_of_unserved = 100 * (len(self.clients_history) / len(self.customers))

        # Final grade
        self.final_grade = (
            weights["waiting"] * self.grade_wait +
            weights["appointment"] * self.grade_appointment +
            weights["unserved"] * self.grade_number_of_unserved
        )

        rows = [
            ("Customer waiting time", self.grade_wait, weights["waiting"]),
            ("Appointment compliance", self.grade_appointment, weights["appointment"]),
            ("Unserved customers", self.grade_number_of_unserved, weights["unserved"]),
        ]

        print("\n--- Performance Summary ---")
        print(f"{'Metric':30} | {'Score':>8} | {'Weight':>6} | {'Weighted':>8}")
        print("-" * 65)

        for name, score, w in rows:
            print(f"{name:30} | {score:8.2f} | {w:6.2f} | {score*w:8.2f}")

        print("-" * 65)
        print(f"{'FINAL GRADE':30} | {self.final_grade:8.2f}")
        