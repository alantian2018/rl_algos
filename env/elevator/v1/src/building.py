import numpy as np
from dataclasses import dataclass


@dataclass
class Person:
    src_floor: int
    target_floor: int
    time_created: float


# need smth to spawn ppl in and encode it i guess.
class Building:
    def __init__(self, num_floors, max_people=None):
        # up, down -> returned to the observation
        # can change later if we wanna do target floor elevators.

        self.num_floors = num_floors
        if max_people is None:
            max_people = num_floors * 2
        self.max_people = max_people
        self.floor_states = np.zeros((num_floors, 2), dtype=bool)
        self.reset()

    def reset(self):
        self.floor_states = np.zeros((self.num_floors, 2), dtype=bool)
        self.people_on_each_floor = np.zeros((self.num_floors, 2), dtype=int)
        self.waiting_people = [[[], []] for _ in range(self.num_floors)]
        self.number_people_waiting = 0
        return self.get_building_state()

    def spawn_people(
        self,
        timestep,
        p=0.01,
    ):

        if self.number_people_waiting >= self.max_people:
            return

        for floor in range(self.num_floors):

            # UP spawn
            if floor < self.num_floors - 1 and np.random.rand() < p:
                target = np.random.randint(floor + 1, self.num_floors)
                self.waiting_people[floor][0].append(Person(floor, target, timestep))
                self.number_people_waiting += 1
            if self.number_people_waiting >= self.max_people:
                break
            # DOWN spawn
            if floor > 0 and np.random.rand() < p:
                target = np.random.randint(0, floor)
                self.waiting_people[floor][1].append(Person(floor, target, timestep))
                self.number_people_waiting += 1
            if self.number_people_waiting >= self.max_people:
                break
        self._refresh_state_from_waiting()

    def _refresh_state_from_waiting(self):
        """Recompute floor_states and people_on_each_floor from waiting_people."""
        for floor in range(self.num_floors):
            self.floor_states[floor][0] = len(self.waiting_people[floor][0]) > 0
            self.people_on_each_floor[floor][0] = len(self.waiting_people[floor][0])
            self.floor_states[floor][1] = len(self.waiting_people[floor][1]) > 0
            self.people_on_each_floor[floor][1] = len(self.waiting_people[floor][1])

    def get_waiting_people(self):
        return self.waiting_people

    def remove_waiting_people(self, num_unloaded):
        self.number_people_waiting -= num_unloaded
        assert self.number_people_waiting >= 0

    def get_building_state(self):
        """Return (floor_states, people_on_each_floor). Always reflects current waiting_people."""
        self._refresh_state_from_waiting()
        return (self.floor_states, self.people_on_each_floor)
