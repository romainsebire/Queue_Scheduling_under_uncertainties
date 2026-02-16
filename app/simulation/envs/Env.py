from app.data.Instance import Instance
from app.data.Scenario import Scenario
from app.domain.Customer import Customer
from app.domain.Server import Server
from app.domain.Appointment import Appointment
from app.simulation.events.Events import Events
from app.simulation.events.CustomerArrivalEvent import CustomerArrivalEvent
from app.simulation.events.CustomerAbandonmentEvent import CustomerAbandonmentEvent
from app.simulation.events.ServerOpeningEvent import ServerOpeningEvent
from app.simulation.events.PendingActivityEvent import PendingActivityEvent
from app.simulation.activity.Service import Service
from app.simulation.activity.Break import Break
import gymnasium as gym
import numpy as np
from enum import Enum, auto
from abc import ABC, abstractmethod
import copy

class Env(gym.Env, ABC):

    class MODE(Enum):
        TEST = auto()
        TRAIN = auto()

    def __init__(self, mode: MODE, instance: Instance = None, scenario: Scenario = None):
        self.mode = mode

        # Data initialization
        if self.mode == Env.MODE.TEST:
            assert instance is not None, "For mode TEST, an instance is required."
            data = instance
        elif self.mode == Env.MODE.TRAIN:
            assert scenario is not None, "For mode TRAIN, a scenario is required."
            self.scenario = scenario
            data = Instance.create(Instance.SourceType.CONFIG,
                                           scenario=self.scenario)
        else:
            raise ValueError("Mode not supported.")

        # Data initializations
        self.c = data.C # number of servers according to Kendall's notation
        self.max_sim_time = data.time_limit if data.time_limit else 630
        self.max_arrival_time = data.max_arrival_time
        self.num_needs = data.num_needs  # number of possible needs
        self.server_unavailability = data.unavailability

        # We intitialize the working server to first server, 
        # it will be changed in _update_next_step()
        #self.current_working_server = 0

        
        self.customers_arrival = self._create_customers_from_steps(data.timeline)
        self.servers = self._build_servers_from_average_matrix(np.array(data.average_matrix, dtype=np.float32))
        self.events = Events()
        self.customer_waiting = dict()
        
        self._add_customer_arrival_events()
        self._add_customer_abandonement_events()

        self.current_server_activity = dict()
        self.planned_server_activity = dict()
        self._get_activities_and_events_from_unavailabilities()

        self.appointments = self._get_appointments_from_list(data.appointments)

        # Action space
        self.action_space = self._get_action_space()
        
        # Observation space
        self.observation_space = self._get_observation_space()

        self.terminated = False
        self.truncated = False
        # General initializations
        self.K = 500 # queue capacity according to Kendall's notation
        self.system_time = 0 # real time (minutes)
        self.steps = 0  # to count the number of RL environment steps (number of decisions made by the agent)

        self.served_clients = 0  # count the total number of served clients
        self.total_service_time = 0.0  # sum of service time for all served clients
        self.total_time_in_system = 0.0  # sum of (waiting + service) time for all served clients
        self.total_waiting_time = 0
        self.max_time_in_system = 0  # to monitore the maximum time spent by a client in the system during an episode
        self.served_clients_info = []
        self.customer_abandonment = 0 # count of customer abandoment
        self.customer_on_service = 0
        self.servers_on_hold = {i: 0 for i in range(self.c)}

        self._update_next_step()

    @abstractmethod
    def _get_action_space(self):
        """
        Get the action space.

        Returns:
            action space compatible with gymnasium
        """
        raise NotImplementedError
    
    @abstractmethod
    def _get_observation_space(self):
        """
        Get the observation space.

        Returns: 
            observation space compatible with gymnasium
        """
        raise NotImplementedError
    
    @abstractmethod
    def _get_obs(self):
        """
        Convert internal state to observation format.

        Returns: 
            obs(np.array)
        """
        raise NotImplementedError
    
    @abstractmethod
    def _get_customer_from_action(self, action) -> Customer:
        """
        Return customer from action.

        Retunrs:
            Customer, or None if invalid action. 
        """   
        raise NotImplementedError    

    @abstractmethod
    def _get_invalid_action_reward(self) -> float: 
        """
        Reward chosen for invalid action.

        Returns:
            reward (float) 
        """  
        raise NotImplementedError 
    
    @abstractmethod
    def _get_valid_reward(self, customer: Customer) -> float:
        """
        Get valid reward.

        Parameters:
            customer (Customer): customer chosen by the action.

        Returns:
            reward (float)
        """
        # ex: return 10
        raise NotImplementedError
    
    @abstractmethod
    def action_masks(self):
        """
        Mask not accepted actions.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _get_hold_action_number(self):
        """
        Get the action to tell the server to hold and not assign a customer.
        """
        raise NotImplementedError

    @classmethod
    def _create_customers_from_steps(self, timeline: list[list[int, int, list, float, float]]):
        """
        Create a list of Customer objects from a matrix of simulation steps.
        """
        customers = {}

        for row in timeline:
            client_id, task, real_service_times, arrival_time, abandonment_time = row

            real_service_times_dict = dict(enumerate(real_service_times))
            customer = Customer(
                id=client_id,
                arrival_time=arrival_time,
                task=task,
                real_service_times=real_service_times_dict,
                abandonment_time=abandonment_time
            )

            customers[client_id] = customer

        return customers

    def _get_activities_and_events_from_unavailabilities(self):
        """
        Create current and pending activities and corresponding events from unavailabilities. 
        """
        for server_id, server in enumerate(self.server_unavailability):
            # No unavailability for this server
            if len(server) == 0:
                continue

            for row in server:
                expected_start, expected_end, break_id = row
                activity = Break(
                    start=None,
                    stop=None,
                    server_id=server_id, 
                    expected_stop=expected_end, 
                    break_type=break_id, 
                    expected_start=expected_start
                )
                # If the break start at the begining of the simulation, 
                # We put the server on break
                if expected_start == 0:
                    activity.start = expected_start
                    activity.stop = expected_end
                    self.current_server_activity[server_id] = activity
                    event = ServerOpeningEvent(activity.stop, activity.server_id)
                # If the break start after time 0, we plan the break
                else: 
                    self.planned_server_activity[activity.id] = activity
                    event = PendingActivityEvent(
                        sim_time=expected_start, 
                        activity_id=activity.id
                    )
                self.events.add(event=event)
       
    @classmethod
    def _build_servers_from_average_matrix(self, service_time_matrix: list[list]):
        """
        Create a dict {server_id: Server} from the average service time matrix.
        """
        num_servers = len(service_time_matrix)
        num_needs = len(service_time_matrix[0])

        servers = {
            s: Server(
                id=s,
                avg_service_time={
                    c: service_time_matrix[s][c]
                    for c in range(num_needs)
                }
            )
            for s in range(num_servers)
        }
        return servers

    @classmethod
    def _get_appointments_from_list(self, appointments_list: list[list[int, int, float]]):
        """
        Initialize appointments from list of appointments. 

        Parameters:
            appointments_list(list[list[int, int, float]]): format: [[[client_id, client_need, appointment_time]], ...]
        
        Returns:
            list[Appointment]
        """
        appointments = dict()
        for row in appointments_list:
            customer_id, task_id, time = row

            appointment = Appointment(
                time = time, 
                customer_id=customer_id, 
                task_id=task_id
            )
            appointments[customer_id] = appointment

        return appointments

    def _add_customer_arrival_events(self):
        """
        Create events for every planned customer arrival.
        """
        for customer in self.customers_arrival.values():
            event = CustomerArrivalEvent(sim_time=customer.arrival_time,
                                         customer_id=customer.id)
            self.events.add(event)

    def _add_customer_abandonement_events(self):
        """
        Create events for every planned customer abandonment.
        """
        for customer in self.customers_arrival.values():
            abandonment_time = customer.abandonment_time

            # If abandonment time = arrival time, the customer never leaves because of waiting
            if abandonment_time != customer.arrival_time:
                event = CustomerAbandonmentEvent(sim_time=customer.abandonment_time,
                                            customer_id=customer.id)
                self.events.add(event)           
    
    def _get_info(self) -> dict[str, any]:
        """
        Compute auxiliary information for debugging. 

        Returns: 
            info under json format       
        """
        avg_service_time = 0
        avg_total_time_in_system = 0
        avg_waiting_time = 0
        if self.served_clients > 0:
            avg_service_time = self.total_service_time / self.served_clients
            avg_total_time_in_system = self.total_time_in_system / self.served_clients
            avg_waiting_time = self.total_waiting_time / self.served_clients
        
        return {
        "served_clients": self.served_clients,
        "avg_service_time": avg_service_time,
        "avg_waiting_time": avg_waiting_time,
        "avg_total_time_in_system": avg_total_time_in_system,
        "worst_time_in_system": self.max_time_in_system,
        "served_clients_info": self.served_clients_info, 
        "action_mask": self.action_masks(), 
        "customer_abandonment": self.customer_abandonment,
        "customers_on_service": self.customer_on_service, 
        "system_time": self.system_time, 
        "queue_length": len(self.customer_waiting), 
        "total_number_of_customers": len(self.customers_arrival), 
        "max_sim_time": self.max_sim_time
        }
        
    def _add_waiting_customer(self, event: CustomerArrivalEvent) -> None:
        """
        Add waiting customer from customer arrival to customer waiting, 
        when an CustomerArrivalEvent happens. 
        """
        # Get customer from customer arrivals
        customer = self.customers_arrival[event.customer_id]

        # Add customer to customer waiting
        self.customer_waiting[customer.id] = customer

    def remove_waiting_customer(self, id: int):
        """
        Remove the selected waiting customer. 
        """
        if not id in self.customer_waiting:
            raise IndexError("Customer waiting key doesn't exist.")

        # Remove the chosen customer
        self.customer_waiting.pop(id)

    def _choose_working_server(self, servers_id: set[int]) -> Server:
        """
        Choose the working server according to server who has less worked.

        Parameters:
            servers_id (set[int]): servers to sort by working time.

        Returns:
            server (Server): server that has work the less
        """
        servers = [self.servers[id] 
                    for id in servers_id 
                    if id in self.servers]
        sorted_servers = sorted(servers, key=lambda s: s.working_time)
        return sorted_servers[0]
    
    def _get_available_servers(self) -> set[int]:
        """
        Get a set of available servers ids.

        Returns:
            set(int): set of available servers id
        """
        available_servers_id = set(self.servers.keys())
        # Check server activity
        for activity in self.current_server_activity.values():
            unavailable_server_id = activity.server_id
            available_servers_id.discard(unavailable_server_id)
        # Check if server on hold
        available_servers_id = {
            sid for sid in available_servers_id
            if self.servers_on_hold[sid] != 1
        }

        return available_servers_id
    
    def _check_existing_possible_service(self) -> set[int]:
        """
        Check if one or more servers can treat one or more waiting customers.

        Returns: 
            set(int): set of compatible servers id
        """
        available_servers_id = self._get_available_servers()

        if len(available_servers_id) == 0:
            return set()
        
        available_servers = [self.servers[id] 
                             for id in available_servers_id 
                             if id in self.servers]
        possible_servers_id = set()
        for customer in self.customer_waiting.values():
            task = customer.task
            for available_server in available_servers:
                if available_server.avg_service_time[task] > 0:
                    possible_servers_id.add(available_server.id)

        return possible_servers_id
    
    def terminate_activity(self, server_id: int):
        """
        Terminate current activity linked to the server_id given.

        Parameters:
            server_id: id of the server the activity must be removed from.
        """
        if server_id not in self.current_server_activity:
            raise RuntimeError(
                f"Server {server_id} has no current activity to remove."
            )
        
        activity = self.current_server_activity.pop(server_id)

        if isinstance(activity, Service):
            working_duration = activity.get_duration()
            self.servers[server_id].increase_working_time(working_duration)
            self.customer_on_service -= 1

    def add_customer(self, customer_id: int):
        """
        Add a customer from customer_arrivals to customer_waiting.

        Parameters: 
            customer_id(int): Id of the new waiting customer.
        """
        if customer_id not in self.customers_arrival:
            raise RuntimeError(
                f"Customer {customer_id} is not planned to arrive."
            )
        self.customer_waiting[customer_id] = self.customers_arrival[customer_id]

    def handle_pending_activity(self, activity_id: int):
        """
        Handle pending activity. 

        Parameters: 
            activity_id (int): Id of the pending activity.
        """
        if activity_id not in self.planned_server_activity:
            raise RuntimeError(
                f"Activity {activity_id} is not pending."
            )
        
        
        pending_activity = self.planned_server_activity[activity_id]

        # Server already in activity
        # We wait for server avaiability to assign the activity
        if pending_activity.server_id in self.current_server_activity: 
            current_activity = self.current_server_activity[pending_activity.server_id]
            event = PendingActivityEvent(
                sim_time=current_activity.stop,
                activity_id=pending_activity.id
            )
        # server is available, the activity starts now
        else:
            pending_activity.start = self.system_time
            delay = self.system_time - pending_activity.expected_start
            pending_activity.stop = pending_activity.expected_stop + delay

            self.planned_server_activity.pop(activity_id)
            self.current_server_activity[pending_activity.server_id] = pending_activity

            event = ServerOpeningEvent(
                sim_time=pending_activity.stop, 
                server_id=pending_activity.server_id
            )
        self.events.add(event=event)

    def _check_truncated(self) -> bool:
        """
        Check if the simulation has been truncated.
        """
        # if system_time inferior to max_sim_time, simulation not finished
        # so it is not truncated
        if self.system_time < self.max_sim_time:
            return False
        
        # There is a customer coming after the end of the simulation
        if self.max_arrival_time > self.max_sim_time:
            return True
        
        # Still customers waiting
        if len(self.customer_waiting) > 0:
            return True
        
        # Servers still in activity
        if len(self.current_server_activity) > 0:
            # The ativity is a customer service
            for activity in self.current_server_activity.values():
                if isinstance(activity, Service):
                    return True
                
        return False
    
    def _calculate_next_sim_time(self) -> set[int]: 
        """
        Play events until a compatibility between server and custtomer is found. 
        Or simulation ends.

        Returns:
            set(int): set of compatible servers id
                if compatible_servers = {}, it is the end of the simulation.
        """
        compatible_servers = {}
        self.servers_on_hold = {i: 0 for i in range(self.c)}

        simulation_ended = False
        compatibility_found = False
        while not simulation_ended and not compatibility_found:
            # No more events but can be remaining waiting customers
            if self.events.is_empty():
                # Every customer has been treated
                if len(self.customer_waiting) == 0:
                    self.truncated = False
                # Still customers that cannot be treated
                else:
                    self.truncated = True
                simulation_ended = True
                self.terminated = True
                continue

            events = self.events.next_batch()
            self.system_time = events[0].sim_time

            if self.system_time >= self.max_sim_time:
                simulation_ended = True
                self.system_time = self.max_sim_time
                if self._check_truncated():
                    self.truncated = True
                else: 
                    self.terminated = True
                continue

            for event in events:
                event.apply(self)
            compatible_servers = self._check_existing_possible_service()
            if len(compatible_servers) > 0:
                compatibility_found = True
        return compatible_servers

    def _update_next_step(self):
        """
        Update state for next_step.
        """
        compatible_servers = self._check_existing_possible_service()

        if len(compatible_servers) == 0:
            compatible_servers = self._calculate_next_sim_time()
            if len(compatible_servers) == 0:
                return 
        
        if len(compatible_servers) == 1:
            server_id = next(iter(compatible_servers))
            self.current_working_server = self.servers[server_id]
        else:
            self.current_working_server = self._choose_working_server(compatible_servers)

    def reset(self, *, seed: int=None, options = None):
        # TODO: use options to modify the configuration
        super().reset(seed=seed)
        
        if self.mode == Env.MODE.TRAIN:
            # Give the reset seed to the config
            self.scenario.seed = seed

            # Create a new instance
            data = Instance.create(Instance.SourceType.CONFIG,
                                           scenario=self.scenario)
            # Data initializations
            self.c = data.C # number of servers according to Kendall's notation
            self.max_sim_time = data.time_limit if data.time_limit else 630
            self.max_arrival_time = data.max_arrival_time
            self.num_needs = data.num_needs  # number of possible needs
            self.server_unavailability = data.unavailability
            self.customers_arrival = self._create_customers_from_steps(data.timeline)
            self.appointments = self._get_appointments_from_list(data.appointments)
            self.servers = self._build_servers_from_average_matrix(np.array(data.average_matrix, dtype=np.float32))
 
        
        
        self.events = Events()
        self._add_customer_arrival_events()
        self._add_customer_abandonement_events()

        self.customer_waiting = dict()
        self.current_server_activity = dict()
        self.planned_server_activity = dict()
        self._get_activities_and_events_from_unavailabilities()


        self.terminated = False
        self.truncated = False
        self.system_time = 0
        self.steps = 0
        self.refused_clients = 0
        self.served_clients = 0
        self.total_service_time = 0.0
        self.total_time_in_system = 0.0
        self.total_waiting_time = 0
        self.max_time_in_system = 0
        self.served_clients_info = []
        self.servers_on_hold = {i: 0 for i in range(self.c)}
        

        self._update_next_step()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def _remove_abandonment_event(self, customer_id: int):
        """
        Remove the CustomerAbandonmentEvent associated to a customer.

        Parameters: 
            customer_id(int): id of the customer
        """
        customer = self.customers_arrival[customer_id]
        event = CustomerAbandonmentEvent(
            sim_time=customer.abandonment_time, 
            customer_id=customer_id
            )
        
        # If abandonment_time == arrival_time, there is no abandonment event.
        if customer.abandonment_time == customer.arrival_time:
            return 
        
        self.events.remove(event=event)  

    def step(self, action):

        self.steps += 1

        if action != self._get_hold_action_number():  # assign the client to a server
            chosen_customer = self._get_customer_from_action(action)
            
            if chosen_customer is None:
                reward = self._get_invalid_action_reward()
            # The chosen task is valid
            else: 
                self.remove_waiting_customer(chosen_customer.id)
                self._remove_abandonment_event(chosen_customer.id)
                self.served_clients += 1
                # get the average time required to satisfy the need using the chosen server (action)
                mean_service = self.current_working_server.avg_service_time[chosen_customer.task]
                real_service = chosen_customer.real_service_times[self.current_working_server.id]

                self.total_service_time += real_service
                self.total_waiting_time += self.system_time - chosen_customer.arrival_time
                total = self.total_waiting_time + real_service
                self.total_time_in_system += total
                if total > self.max_time_in_system:
                    self.max_time_in_system = total
                
                assign_time = self.system_time
                end_time = assign_time + real_service

                self.served_clients_info.append({
                    'server': self.current_working_server.id,
                    'arrival': chosen_customer.arrival_time,
                    'start': assign_time,
                    'end': end_time,
                    'client': chosen_customer.id,
                    'class': chosen_customer.task,
                    'estimated_proc_time': mean_service,
                    'real_proc_time': end_time - assign_time,
                })

                # Bonus: 
                reward = self._get_valid_reward(chosen_customer)
                
                # Add activity
                self.current_server_activity[self.current_working_server.id] = Service(
                    start=self.system_time, 
                    stop=end_time, 
                    server_id=self.current_working_server.id,
                    customer_id=chosen_customer.id,
                    task_id=chosen_customer.task,
                    expected_stop=self.system_time + mean_service
                    )
                
                # Add event to reopen the server at the end of the activity
                event = ServerOpeningEvent(
                    sim_time=end_time, 
                    server_id=self.current_working_server.id
                )
                self.events.add(event=event)
                self.customer_on_service += 1

                # If an appointment, update appointment service time for evaluation
                if chosen_customer.id in self.appointments:
                    self.appointments[chosen_customer.id].set_service_time(self.system_time)


        else:  # HOLD action 
            self.servers_on_hold[self.current_working_server.id] = 1
            reward = -1

        self._update_next_step()

        next_observation = self._get_obs()

        info = self._get_info()
        return next_observation, reward, self.terminated, self.truncated, info
    
    def increase_customer_abandonment_count(self):
        """
        Increase the count of customer abandonment. 
        """
        self.customer_abandonment += 1

    def _get_state(self):
        """
        Get current state, usable for observations.
        """
        waiting_customers = copy.deepcopy(self.customer_waiting)

        for c in waiting_customers.values():
            c.real_service_times = None
            c.abandonment_time = None


        appointments = copy.deepcopy(self.appointments)

        servers = copy.deepcopy(self.servers)

        end_of_service = {
            server_id: self.current_server_activity.get(server_id, None).expected_stop
            if server_id in self.current_server_activity
            else 0
            for server_id in range(self.c)
        }

        selected_server_id = self.current_working_server.id

        current_sim_time = self.system_time

        return waiting_customers, appointments, servers, end_of_service, selected_server_id, current_sim_time


            



            
            
            
            