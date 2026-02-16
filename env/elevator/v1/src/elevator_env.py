import gymnasium as gym
import numpy as np
from .building import Building
from .elevator_v1 import ElevatorWrapper


class ElevatorEnv(gym.Env):
    def __init__(self, num_elevators, num_floors, max_people=None):
        super().__init__()
        self.building = Building(num_floors, max_people)
        self.elevator_wrapper = ElevatorWrapper(num_elevators, num_floors, max_people)
        self.num_floors = num_floors
        self.max_people = max_people
        self.num_elevators = num_elevators

    def step(self, actions, timestep):
        # num people waiting -= num_loaded eventually, do not forget this plz or
        # nobody new will spawn
        raise NotImplementedError("Step not implemented")

    def reset(self):
        floor_state, people_on_each_floor = self.building.reset()
        elevator_obs = self.elevator_wrapper.reset()
        return np.concatenate([floor_state.flatten(), elevator_obs])

    def render(self, mode="human"):
        # building.get_state => returns floor_states (ie up/down for each floor) and people on each floor

        # we need both to render the ppl and the elevators.
        pass
