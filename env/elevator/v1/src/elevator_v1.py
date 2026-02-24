# naively ignore physics
# no marl for now

import gymnasium
from enum import IntEnum
import numpy as np


# action space = [up, down, idle] * num elevators
# one time step => do the step and ignore momentum
class ElevatorActions(IntEnum):
    IDLE = 0
    UP = 1
    DOWN = -1


# observation space:
# 1. position of elevators currently
# 2. If up/down status is triggered at each floor. Ie [ up_botton_pressed, down_button_pressed ] * num_floors
# 3. time since each button has been pressed for? maybe? ill see if this is important


class Elevator:

    def __init__(
        self,
        max_floor,
        start_floor=0,
    ):
        self.max_floor = max_floor
        self.start_floor = start_floor
        self.reset()

    def load_people(self, waiting_people, action):
        if action == ElevatorActions.UP:
            self.carrying_people.extend(waiting_people[self.current_floor][0])

            # delete it from waiting_people
            waiting_people[self.current_floor][0] = []
        elif action == ElevatorActions.DOWN:
            self.carrying_people.extend(waiting_people[self.current_floor][1])
            waiting_people[self.current_floor][1] = []
        elif action == ElevatorActions.IDLE:
            pass
        else:
            raise ValueError(f"Invalid action: {action}")

    def unload_people(self):
        num_unloaded = 0
        for person in self.carrying_people:
            if person.target_floor == self.current_floor:
                self.carrying_people.remove(person)
                num_unloaded += 1
        return num_unloaded

    def reset(self):
        self.current_floor = self.start_floor
        self.last_action = 0
        self.carrying_people = []
        self.target_floors = np.zeros(self.max_floor)
        return self.get_state()

    def get_state(self):
        # concatenate one hot current floor, target floors, last action one hot
        current_floor_one_hot = np.zeros(self.max_floor)
        current_floor_one_hot[self.current_floor] = 1
        target_floors_one_hot = np.zeros(self.max_floor)
        for person in self.carrying_people:
            target_floors_one_hot[person.target_floor] = 1
        last_action_one_hot = np.zeros(3)
        last_action_one_hot[self.last_action + 1] = 1
        return np.concatenate(
            [current_floor_one_hot, target_floors_one_hot, last_action_one_hot]
        )

    def step(self, action, waiting_people, timestep):
        # subtract one from action since it is [0, 1, 2] -> [-1, 0, 1] -> [down, idle, up]
        action -= 1

        assert action in [-1, 0, 1], "Invalid action"

        self.last_action = action

        # unload ppl here
        num_unloaded = self.unload_people()
        # load ppl here
        self.load_people(waiting_people, action)
        # update target floors
        elevator_waiting_times = []
        self.target_floors = np.zeros(self.max_floor)
        for person in self.carrying_people:
            elevator_waiting_times.append(timestep - person.time_created)
            self.target_floors[person.target_floor] = 1

        did_invalid_action = False
        # move elevator
        if action == ElevatorActions.UP and self.current_floor == self.max_floor - 1:
            did_invalid_action = True
        elif action == ElevatorActions.DOWN and self.current_floor == 0:
            did_invalid_action = True
        else:
            self.current_floor += action

        obs = self.get_state()

        return obs, num_unloaded, did_invalid_action, elevator_waiting_times


class ElevatorWrapper:
    # yes you can do this vectorized for concurrency but i think this is easier to understand and scale
    def __init__(self, max_elevators, max_floor, start_floor=0):
        self.elevators: list[Elevator] = []
        for _ in range(max_elevators):
            self.elevators.append(Elevator(max_floor, start_floor))

    def step(self, actions, waiting_people, timestep):
        
        assert len(actions) == len(self.elevators), "Invalid number of actions"
        num_unloaded_list = []
        elevator_obs = []
        did_invalid_actions = []
        elevator_waiting_times_list = []
        for c, action in enumerate(actions):
            obs, n_unloaded, did_invalid_action, e_waiting_times = self.elevators[
                c
            ].step(action, waiting_people, timestep)

            elevator_obs.append(obs)
            num_unloaded_list.append(n_unloaded)
            did_invalid_actions.append(did_invalid_action)
            elevator_waiting_times_list.extend(e_waiting_times)

        reward = self.calculate_reward(
            num_unloaded_list,
            any(did_invalid_actions),
            waiting_people,
            elevator_waiting_times_list,
        )
        info = {"did_invalid_actions": did_invalid_actions}
        if elevator_waiting_times_list:
            info["mean_elevator_waiting_time"] = sum(elevator_waiting_times_list) / len(elevator_waiting_times_list)
            info["max_elevator_waiting_time"] = max(elevator_waiting_times_list)
            info["min_elevator_waiting_time"] = min(elevator_waiting_times_list)

        return (
            np.array(elevator_obs).flatten(),
            reward,
            sum(num_unloaded_list),
            info,
        )

    def reset(self):
        elevator_obs = []
        for elevator in self.elevators:
            obs = elevator.reset()
            elevator_obs.append(obs)
        return np.array(elevator_obs).flatten()

    def calculate_reward(
        self, num_unloaded, did_invalid_action, waiting_people, elevator_waiting_times
    ):
        reward = sum(num_unloaded)
        reward -= 10 if did_invalid_action else 0
        # add timestep penalty

        reward -= (
            sum(elevator_waiting_times) * 0.01
        )
        for floor in waiting_people:
            for direction in floor:
                reward -= len(direction) * 0.01
        return reward 
        # elevator reward function
        # add the people waiting penalty
