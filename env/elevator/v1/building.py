import numpy as np
from dataclasses import dataclass
import gymnasium as gym
from elevator_v1 import ElevatorWrapper
 





@dataclass
class Person:
    src_floor : int
    target_floor : int
    time_created : float
 
class ElevatorEnv(gym.Env):
    def __init__(self, num_elevators, num_floors, max_people = None):
        super().__init__()
        self.building = Building(num_floors, max_people)
        self.elevator_wrapper = ElevatorWrapper(num_elevators, num_floors, max_people)
        self.num_floors = num_floors
        self.max_people = max_people
        self.num_elevators = num_elevators


# need smth to spawn ppl in and encode it i guess.
class Building:
    def __init__(self, num_floors, max_people = None):
                                         # up, down -> returned to the observation
        # can change later if we wanna do target floor elevators.
        self.floor_states = np.zeros((num_floors, 2), dtype=bool)
                                       #num ppl up, num ppl down
        self.people_on_each_floor =np.zeros((num_floors, 2), dtype=int)

        self.num_floors = num_floors
        if max_people is None:
            max_people = num_floors * 2
        self.max_people = max_people
        self.waiting_people = [
            [[], []] for _ in range(num_floors)
        ]
        self.number_people_waiting = 0
    
    
    def spawn_people(self,  timestep, p=0.01,):
    
        if len(self.number_people_waiting) >= self.max_people:
            return

        for floor in range(self.num_floors):

            # UP spawn
            if floor < self.num_floors - 1 and np.random.rand() < p:
                target = np.random.randint(floor + 1, self.num_floors)
                self.waiting_people[floor][0].append(Person(floor, target, timestep))
                self.number_people_waiting += 1
            
            # DOWN spawn
            if floor > 0 and np.random.rand() < p:
                target = np.random.randint(0, floor)
                self.waiting_people[floor][1].append(Person(floor, target, timestep))
                self.number_people_waiting += 1
        # scan thru waiting people and update floor_states if there are people waiting
        for floor in range(self.num_floors):
            if len(self.waiting_people[floor][0]) > 0:
                self.floor_states[floor][0] = True
                self.people_on_each_floor[floor][0] = len(self.waiting_people[floor][0])
            if len(self.waiting_people[floor][1]) > 0:
                self.floor_states[floor][1] = True
                self.people_on_each_floor[floor][1] = len(self.waiting_people[floor][1])
       


    def get_building_state(self):
        return (
            self.floor_states,
            self.people_on_each_floor
        )
    
  