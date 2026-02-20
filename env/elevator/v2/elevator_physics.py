## TODO fix this later but try naive step of env for now...
## Started implementing this than realized its way to difficult haha

import gymnasium as gym
from enum import Enum
import numpy as np


class ElevatorMovements(Enum):
    UP = 1
    DOWN = -1
    IDLE = 0


class ElevatorTimes(Enum):
    MOVE_ONE_FLOOR = 2  # 2 seconds to move one floor

    # TODO ill add this later this seems kinda like a hassle to think about rn
    # STOP_OR_ACCELERATE = 1 # one second to get up to speed?

    # TODO we'll just assume instant loading for now
    # OPEN_AND_CLOSE_DOORS = 1.5 # one and a half to open and close da door, can then add time it takes to load a person on
    # TIME_TO_LOAD_A_PERSON = 0.2


INF = 1e9


class Person:
    def __init__(self, start_floor, target_floor):
        self.start_floor = start_floor
        self.target_floor = target_floor
        self.time_elapsed = 0

    def add_time(self, time):
        self.time_elapsed += time

    def get_waiting_time(self):
        return self.time_elapsed


class Elevator:
    def __init__(self, floors):
        self.total_floors = floors
        self.current_floor = 1
        self.target_floor = 1
        self.carrying_people = 0

        # destinations where ppl wanna go
        # this will be one shifted, hopefully i dont forget that (ie floor 1-> index 0)
        # This is only for ppl inside the elevator!
        self.target_destinations = [0] * floors

        self.current_state = ElevatorMovements.IDLE

    # so actions can be
    def step(self, new_action, time_elapsed):
        # action will be a "go to floor x" or not move at all. (0)
        #  we'll just open the doors if there is at least one person at that floor.

        # first calculate what was going on in the time elapsed.
        floors_to_be_covered = self.target_floor - self.current_floor
        time_needed = self._time_to_get_to_target_floor()
        floors_covered = time_elapsed / time_needed * floors_to_be_covered
        # lets just make sure these are the same sign
        assert self.current_state * floors_covered >= 0
        self.current_floor += floors_covered
        assert self.current_floor >= 1
        if self.current_floor == self.target_floor:
            self.current_state = ElevatorMovements.IDLE

        # great, now update the action...
        if new_action > 0:
            self.target_floor = new_action
            if self.target_floor < self.current_floor:
                self.current_state = ElevatorMovements.DOWN
            elif self.target_floor > self.current_floor:
                self.current_state = ElevatorMovements.UP
            else:
                self.current_state = ElevatorMovements.IDLE

    def load_people(self, cur_floor, people: list[Person]):
        # technically cur_floor is not needed but just to be safe...
        assert cur_floor == self.current_floor
        for person in people:
            pass

    def get_next_event(self):
        # when will the elevator get to the target floor?
        if self.current_floor != self.target_floor:
            assert self.current_state != ElevatorMovements.IDLE
            time_needed = self._time_to_get_to_target_floor()
            return time_needed
        # this elevator is already at said floor, so it will never trigger an event (we rely on passengers arriving)
        return INF

    def _time_to_get_to_target_floor(self):
        return (
            np.abs(self.target_floor - self.current_floor)
            * ElevatorTimes.MOVE_ONE_FLOOR
        )


class Building(gym.Env):
    def __init__(
        self, elevators, floors, button_inside_elevator=True, render_mode=None
    ):
        super().__init__()
