import gymnasium as gym
import numpy as np
import cv2

from .building import Building
from .elevator_v1 import ElevatorWrapper
from .render import render_building


class ElevatorEnv(gym.Env):
    def __init__(
        self,
        num_elevators,
        num_floors,
        max_steps=10_000,
        max_people=None,
        spawn_rate=0.01,
        render_mode="rgb_array",
    ):
        super().__init__()
        self.building = Building(num_floors, max_people)
        self.elevator_wrapper = ElevatorWrapper(num_elevators, num_floors)
        self.num_floors = num_floors
        self.max_people = max_people
        self.num_elevators = num_elevators
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.spawn_rate = spawn_rate
        self.reset()

    def step(self, actions):
        if actions.ndim == 0:
            actions = np.array([actions])

        # num people waiting -= num_loaded eventually, do not forget this plz or
        # nobody new will spawn
        # loop should be

        # when to spawn people? at the end of beginning? probably the end
        # 1. step elevators
        # 2.
        timestep = self.current_step
        waiting_people = self.building.get_waiting_people()
        elevator_obs, reward, num_unloaded, info = self.elevator_wrapper.step(
            actions, waiting_people, timestep
        )
        # if unloaded people remove it
        self.building.remove_waiting_people(num_unloaded)

        self.building.spawn_people(timestep, p=self.spawn_rate)
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # get building obs and concat with elevator obs
        floor_states, people_on_each_floor = self.building.get_building_state()
        obs = np.concatenate([floor_states.flatten(), elevator_obs])
        return obs, reward, done, False, info

    def reset(self):
        floor_state, people_on_each_floor = self.building.reset()
        elevator_obs = self.elevator_wrapper.reset()
        self.current_step = 0
        return np.concatenate([floor_state.flatten(), elevator_obs]), {}

    def render(self, render_mode=None):
        render_mode = render_mode or self.render_mode
        assert render_mode in ["human", "rgb_array"], "Invalid render mode"
        floor_states, people_on_each_floor = self.building.get_building_state()
        img = render_building(
            self.num_floors,
            self.num_elevators,
            floor_states,
            people_on_each_floor,
            self.elevator_wrapper.elevators,
            self.current_step,
        )
        if render_mode == "human":
            cv2.imshow("Elevator Env", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        elif render_mode == "rgb_array":
            return img

    def close(self):
        if self.render_mode == "human":
            cv2.destroyAllWindows()
