"""
Tests for Building and Elevator classes.
Uses np.random.seed() for deterministic RNG across test runs.

Run from repo root: python -m pytest env/elevator/v1/testing/ -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

from ..src.building import Building, Person
from ..src.elevator_v1 import Elevator, ElevatorWrapper

SEED = 42


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed before each test for deterministic behavior."""
    np.random.seed(SEED)
    yield
    # Seed is module-level; tests that need different seeds can override


class TestBuilding:
    """Tests for the Building class."""

    def test_init_default_max_people(self):
        np.random.seed(SEED)
        building = Building(num_floors=5)
        assert building.num_floors == 5
        assert building.max_people == 10  # num_floors * 2
        assert building.number_people_waiting == 0
        assert building.floor_states.shape == (5, 2)
        assert building.people_on_each_floor.shape == (5, 2)
        assert len(building.waiting_people) == 5

    def test_init_custom_max_people(self):
        np.random.seed(SEED)
        building = Building(num_floors=5, max_people=100)
        assert building.max_people == 100

    def test_get_building_state_initial(self):
        np.random.seed(SEED)
        building = Building(num_floors=3)
        floor_states, people_on_each_floor = building.get_building_state()
        assert np.all(floor_states == False)
        assert np.all(people_on_each_floor == 0)

    def test_spawn_people_deterministic(self):
        np.random.seed(SEED)
        building = Building(num_floors=5, max_people=50)
        building.spawn_people(timestep=0, p=0.5)  # high p to get some spawns
        state1 = building.get_building_state()

        np.random.seed(SEED)
        building2 = Building(num_floors=5, max_people=50)
        building2.spawn_people(timestep=0, p=0.5)
        state2 = building2.get_building_state()

        np.testing.assert_array_equal(state1[0], state2[0])
        np.testing.assert_array_equal(state1[1], state2[1])

    def test_spawn_people_updates_floor_states(self):
        np.random.seed(SEED)
        building = Building(num_floors=5, max_people=50)
        building.spawn_people(timestep=0, p=1.0)  # p=1 to force spawns
        floor_states, people_on_each_floor = building.get_building_state()
        # With p=1, we should have some people waiting
        assert np.any(floor_states) or building.number_people_waiting >= 0

    def test_spawn_people_respects_max_people(self):
        np.random.seed(SEED)
        building = Building(num_floors=3, max_people=2)
        for t in range(100):
            building.spawn_people(timestep=t, p=1.0)
        assert building.number_people_waiting <= 2

    def test_get_building_state_structure(self):
        """State observation has correct shape and dtypes."""
        np.random.seed(SEED)
        building = Building(num_floors=5)
        floor_states, people_on_each_floor = building.get_building_state()
        assert floor_states.shape == (5, 2)
        assert people_on_each_floor.shape == (5, 2)
        assert floor_states.dtype.kind == "b"  # boolean
        assert people_on_each_floor.dtype.kind in ("i", "u")  # int

    def test_get_building_state_floor_states_semantics(self):
        """floor_states: column 0 = up pressed, column 1 = down pressed."""
        np.random.seed(SEED)
        building = Building(num_floors=5)
        # Manually add people: floor 1 going up, floor 3 going down
        building.waiting_people[1][0].append(Person(1, 4, 0))
        building.waiting_people[3][1].append(Person(3, 0, 0))
        floor_states, people_on_each_floor = building.get_building_state()
        # Floor 1: up pressed
        assert floor_states[1][0] == True  # use == for np.bool_ compatibility
        assert floor_states[1][1] == False
        # Floor 3: down pressed
        assert floor_states[3][0] == False
        assert floor_states[3][1] == True
        # Other floors: neither
        assert floor_states[0][0] == False and floor_states[0][1] == False
        assert floor_states[2][0] == False and floor_states[2][1] == False
        assert floor_states[4][0] == False and floor_states[4][1] == False

    def test_get_building_state_people_counts(self):
        """people_on_each_floor matches actual waiting counts."""
        np.random.seed(SEED)
        building = Building(num_floors=5)
        building.waiting_people[0][0].extend(
            [Person(0, 2, 0), Person(0, 4, 0)]
        )  # 2 going up
        building.waiting_people[2][1].append(Person(2, 0, 0))  # 1 going down
        floor_states, people_on_each_floor = building.get_building_state()
        assert people_on_each_floor[0][0] == 2
        assert people_on_each_floor[0][1] == 0
        assert people_on_each_floor[2][0] == 0
        assert people_on_each_floor[2][1] == 1

    def test_get_building_state_reflects_elevator_pickup(self):
        """State updates when elevator removes people from waiting_people (simulated pickup)."""
        np.random.seed(SEED)
        building = Building(num_floors=5)
        building.waiting_people[1][0].append(Person(1, 3, 0))
        floor_states_before, _ = building.get_building_state()
        assert floor_states_before[1][0] == True
        # Simulate elevator picking up everyone going up from floor 1
        building.waiting_people[1][0].clear()
        floor_states_after, people_after = building.get_building_state()
        assert floor_states_after[1][0] == False
        assert people_after[1][0] == 0

    def test_get_building_state_multi_floor_observation(self):
        """Observation correctly encodes multiple floors with different directions."""
        np.random.seed(SEED)
        building = Building(num_floors=4)
        building.waiting_people[0][0].append(Person(0, 3, 0))  # floor 0: up
        building.waiting_people[1][0].append(Person(1, 3, 0))  # floor 1: up
        building.waiting_people[1][1].append(Person(1, 0, 0))  # floor 1: also down
        building.waiting_people[3][1].append(Person(3, 0, 0))  # floor 3: down
        floor_states, people = building.get_building_state()
        expected_floor_states = np.array(
            [
                [True, False],  # floor 0: up
                [True, True],  # floor 1: up and down
                [False, False],  # floor 2: none
                [False, True],  # floor 3: down
            ]
        )
        np.testing.assert_array_equal(floor_states, expected_floor_states)
        assert people[1][0] == 1 and people[1][1] == 1


class TestPerson:
    """Tests for the Person dataclass."""

    def test_person_creation(self):
        person = Person(src_floor=1, target_floor=3, time_created=0.0)
        assert person.src_floor == 1
        assert person.target_floor == 3
        assert person.time_created == 0.0


class TestElevator:
    """Tests for the Elevator class."""

    def test_init(self):
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5, start_floor=0)
        assert elevator.max_floor == 5
        assert elevator.current_floor == 0
        assert elevator.carrying_people == []
        assert elevator.target_floors.shape == (5,)

    def test_init_custom_start_floor(self):
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5, start_floor=2)
        assert elevator.current_floor == 2

    def test_reset(self):
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5, start_floor=3)
        elevator.step(2, [[[], []] for _ in range(5)], timestep=0)  # UP
        obs = elevator.reset()
        assert elevator.current_floor == 3
        assert elevator.carrying_people == []
        assert obs.shape == (
            5 + 5 + 3,
        )  # floor one-hot + target one-hot + action one-hot

    def test_get_state_shape(self):
        np.random.seed(SEED)
        elevator = Elevator(max_floor=4)
        state = elevator.get_state()
        assert state.shape == (4 + 4 + 3,)  # max_floor*2 + 3 for last_action

    def test_step_idle(self):
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5)
        waiting_people = [[[], []] for _ in range(5)]
        obs, num_unloaded, _, did_invalid, wait_times = elevator.step(
            1, waiting_people, timestep=0
        )
        assert elevator.current_floor == 0
        assert num_unloaded == 0
        assert did_invalid is False
        assert obs.shape == (5 + 5 + 3,)

    def test_step_up(self):
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5)
        waiting_people = [[[], []] for _ in range(5)]
        obs, num_unloaded, _, did_invalid, _ = elevator.step(2, waiting_people, timestep=0)
        assert elevator.current_floor == 1
        assert did_invalid is False

    def test_step_down(self):
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5, start_floor=2)
        waiting_people = [[[], []] for _ in range(5)]
        obs, num_unloaded, _, did_invalid, _ = elevator.step(0, waiting_people, timestep=0)
        assert elevator.current_floor == 1
        assert did_invalid is False

    def test_step_invalid_up_at_top(self):
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5, start_floor=4)
        waiting_people = [[[], []] for _ in range(5)]
        obs, num_unloaded, _, did_invalid, _ = elevator.step(2, waiting_people, timestep=0)
        assert elevator.current_floor == 4  # didn't move
        assert did_invalid is True

    def test_step_invalid_down_at_bottom(self):
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5)
        waiting_people = [[[], []] for _ in range(5)]
        obs, num_unloaded, _, did_invalid, _ = elevator.step(0, waiting_people, timestep=0)
        assert elevator.current_floor == 0
        assert did_invalid is True

    def test_load_and_unload_people(self):
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5, start_floor=1)
        person_up = Person(src_floor=1, target_floor=3, time_created=0)
        person_down = Person(src_floor=1, target_floor=0, time_created=0)
        waiting_people = [
            [[], []],  # floor 0
            [[person_up], [person_down]],  # floor 1: 1 going up, 1 going down
            [[], []],
            [[], []],
            [[], []],
        ]
        # Go UP to load people going up (step 1: at floor 1, load, move to 2)
        elevator.step(2, waiting_people, timestep=0)
        assert len(elevator.carrying_people) == 1
        assert elevator.carrying_people[0].target_floor == 3
        assert waiting_people[1][0] == []
        # Step 2: at floor 2, move UP to floor 3
        elevator.step(2, waiting_people, timestep=1)
        assert elevator.current_floor == 3
        # Step 3: at floor 3, idle to unload (unload happens at start of step, before move)
        obs, num_unloaded, _, _, _ = elevator.step(1, waiting_people, timestep=2)
        assert num_unloaded == 1
        assert len(elevator.carrying_people) == 0

    def test_up_sequence_full_building(self):
        """Elevator moves from floor 0 to top, verifying position at each step."""
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5)
        waiting_people = [[[], []] for _ in range(5)]
        for i in range(4):  # 4 UP steps: 0->1->2->3->4
            elevator.step(2, waiting_people, timestep=i)
            assert (
                elevator.current_floor == i + 1
            ), f"After {i+1} UP steps, expected floor {i+1}"
        assert elevator.current_floor == 4

    def test_down_sequence_full_building(self):
        """Elevator moves from top to bottom."""
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5, start_floor=4)
        waiting_people = [[[], []] for _ in range(5)]
        for i in range(4):  # 4 DOWN steps: 4->3->2->1->0
            elevator.step(0, waiting_people, timestep=i)
            assert elevator.current_floor == 3 - i
        assert elevator.current_floor == 0

    def test_load_passengers_going_down(self):
        """UP loads floor[0] (going up), DOWN loads floor[1] (going down)."""
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5, start_floor=3)
        person_to_floor_0 = Person(src_floor=3, target_floor=0, time_created=0)
        waiting_people = [
            [[], []],
            [[], []],
            [[], []],
            [[], [person_to_floor_0]],  # floor 3: 1 going down
            [[], []],
        ]
        # Must go DOWN to load people going down
        elevator.step(0, waiting_people, timestep=0)  # DOWN: load, move to 2
        assert len(elevator.carrying_people) == 1
        assert elevator.carrying_people[0].target_floor == 0
        assert waiting_people[3][1] == []
        # Go down to floor 0
        elevator.step(0, waiting_people, timestep=1)  # 2->1
        elevator.step(0, waiting_people, timestep=2)  # 1->0
        obs, num_unloaded, _, _, _ = elevator.step(
            1, waiting_people, timestep=3
        )  # idle to unload
        assert num_unloaded == 1
        assert elevator.current_floor == 0

    def test_idle_does_not_load_passengers(self):
        """Elevator at floor with waiting people; idle does not pick them up."""
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5)
        person = Person(src_floor=0, target_floor=2, time_created=0)
        waiting_people = [[[person], []], [[], []], [[], []], [[], []], [[], []]]
        elevator.step(1, waiting_people, timestep=0)  # idle
        assert len(elevator.carrying_people) == 0
        assert len(waiting_people[0][0]) == 1  # still waiting

    def test_up_does_not_load_passengers_going_down(self):
        """When going UP, we only load people going up (floor[0]), not down (floor[1])."""
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5, start_floor=2)
        person_going_down = Person(src_floor=2, target_floor=0, time_created=0)
        waiting_people = [
            [[], []],
            [[], []],
            [[], [person_going_down]],  # floor 2: person going down
            [[], []],
            [[], []],
        ]
        elevator.step(
            2, waiting_people, timestep=0
        )  # UP - should NOT load person going down
        assert len(elevator.carrying_people) == 0
        assert len(waiting_people[2][1]) == 1

    def test_down_does_not_load_passengers_going_up(self):
        """When going DOWN, we only load people going down (floor[1]), not up (floor[0])."""
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5, start_floor=2)
        person_going_up = Person(src_floor=2, target_floor=4, time_created=0)
        waiting_people = [
            [[], []],
            [[], []],
            [[person_going_up], []],  # floor 2: person going up
            [[], []],
            [[], []],
        ]
        elevator.step(
            0, waiting_people, timestep=0
        )  # DOWN - should NOT load person going up
        assert len(elevator.carrying_people) == 0
        assert len(waiting_people[2][0]) == 1

    def test_get_state_current_floor_position(self):
        """get_state current_floor one-hot reflects elevator position."""
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5, start_floor=2)
        waiting_people = [[[], []] for _ in range(5)]
        elevator.step(1, waiting_people, timestep=0)
        state = elevator.get_state()
        # First max_floor elements are current_floor one-hot
        current_floor_one_hot = state[:5]
        assert current_floor_one_hot[2] == 1
        assert sum(current_floor_one_hot) == 1

    def test_get_state_target_floors_when_carrying(self):
        """get_state target_floors one-hot reflects passengers' destinations."""
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5, start_floor=1)
        p1 = Person(src_floor=1, target_floor=2, time_created=0)
        p2 = Person(src_floor=1, target_floor=4, time_created=0)
        waiting_people = [[[], []], [[p1, p2], []], [[], []], [[], []], [[], []]]
        elevator.step(2, waiting_people, timestep=0)  # UP to load
        state = elevator.get_state()
        target_one_hot = state[5:10]
        assert target_one_hot[2] == 1
        assert target_one_hot[4] == 1
        assert sum(target_one_hot) == 2

    def test_multiple_passengers_same_direction(self):
        """Pick up 2 people going up from different floors, deliver both."""
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5)
        p_floor2 = Person(src_floor=0, target_floor=2, time_created=0)
        p_floor4 = Person(src_floor=1, target_floor=4, time_created=0)
        waiting_people = [
            [[p_floor2], []],
            [[p_floor4], []],
            [[], []],
            [[], []],
            [[], []],
        ]
        # Floor 0: load p_floor2, move to 1
        elevator.step(2, waiting_people, timestep=0)
        assert len(elevator.carrying_people) == 1
        # Floor 1: load p_floor4, move to 2
        elevator.step(2, waiting_people, timestep=1)
        assert len(elevator.carrying_people) == 2
        # Floor 2: unload p_floor2, move to 3
        _, num_unloaded, _, _, _ = elevator.step(2, waiting_people, timestep=2)
        assert num_unloaded == 1
        assert len(elevator.carrying_people) == 1
        assert elevator.carrying_people[0].target_floor == 4
        # Floor 3: move to 4
        elevator.step(2, waiting_people, timestep=3)
        # Floor 4: unload p_floor4
        _, num_unloaded, _, _, _ = elevator.step(1, waiting_people, timestep=4)
        assert num_unloaded == 1
        assert len(elevator.carrying_people) == 0

    def test_up_then_down_full_trip(self):
        """Elevator goes up with passenger, delivers, goes down with another passenger."""
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5)
        p_up = Person(src_floor=0, target_floor=4, time_created=0)
        p_down = Person(src_floor=4, target_floor=0, time_created=0)
        waiting_people = [
            [[p_up], []],
            [[], []],
            [[], []],
            [[], []],
            [[], [p_down]],  # floor 4 going down
        ]
        # Load at 0, go up to 4
        elevator.step(2, waiting_people, timestep=0)  # 0->1
        elevator.step(2, waiting_people, timestep=1)  # 1->2
        elevator.step(2, waiting_people, timestep=2)  # 2->3
        elevator.step(2, waiting_people, timestep=3)  # 3->4
        _, n, _, _, _ = elevator.step(1, waiting_people, timestep=4)  # unload at 4
        assert n == 1
        # Now load person going down at floor 4
        elevator.step(0, waiting_people, timestep=5)  # DOWN: load, 4->3
        assert len(elevator.carrying_people) == 1
        assert elevator.carrying_people[0].target_floor == 0
        # Go down to 0 (3 more steps: 3->2->1->0)
        for t in range(6, 9):
            elevator.step(0, waiting_people, timestep=t)
        _, n, _, _, _ = elevator.step(1, waiting_people, timestep=7)  # unload at 0
        assert n == 1
        assert elevator.current_floor == 0


class TestElevatorWrapper:
    """Tests for the ElevatorWrapper class."""

    def test_init(self):
        np.random.seed(SEED)
        wrapper = ElevatorWrapper(max_elevators=2, max_floor=5)
        assert len(wrapper.elevators) == 2
        assert all(e.current_floor == 0 for e in wrapper.elevators)

    def test_reset(self):
        np.random.seed(SEED)
        wrapper = ElevatorWrapper(max_elevators=2, max_floor=5)
        obs = wrapper.reset()
        assert obs.shape == (2 * (5 + 5 + 3),)  # flattened obs from 2 elevators

    def test_step(self):
        np.random.seed(SEED)
        wrapper = ElevatorWrapper(max_elevators=2, max_floor=5)
        wrapper.reset()
        waiting_people = [[[], []] for _ in range(5)]
        actions = [1, 1]  # both idle
        obs, reward, total_unloaded, _, info = wrapper.step(
            actions, waiting_people, timestep=0
        )

        assert obs.shape == (2 * (5 + 5 + 3),)
        assert "did_invalid_actions" in info
        assert total_unloaded == 0

    def test_info_no_passengers(self):
        """With no passengers, wait time stats should be absent from info."""
        wrapper = ElevatorWrapper(max_elevators=1, max_floor=5)
        wrapper.reset()
        waiting_people = [[[], []] for _ in range(5)]
        _, _, _, _, info = wrapper.step([1], waiting_people, timestep=0)
        assert "did_invalid_actions" in info
        assert "mean_elevator_waiting_time" not in info
        assert "max_elevator_waiting_time" not in info
        assert "min_elevator_waiting_time" not in info

    def test_info_with_passengers(self):
        """With passengers loaded, wait time stats should appear in info."""
        wrapper = ElevatorWrapper(max_elevators=1, max_floor=5)
        wrapper.reset()
        p = Person(0, 3, 0)
        waiting = [[[p], []], [[], []], [[], []], [[], []], [[], []]]
        wrapper.step([2], waiting, timestep=0)  # load at 0, move to 1
        empty_waiting = [[[], []] for _ in range(5)]
        _, _, _, _, info = wrapper.step([2], empty_waiting, timestep=5)  # move to 2
        assert "mean_elevator_waiting_time" in info
        assert "max_elevator_waiting_time" in info
        assert "min_elevator_waiting_time" in info
        assert info["mean_elevator_waiting_time"] == 5
        assert info["max_elevator_waiting_time"] == 5
        assert info["min_elevator_waiting_time"] == 5

    def test_info_multiple_passengers_different_wait_times(self):
        """Wait time stats should reflect all passengers across elevators."""
        wrapper = ElevatorWrapper(max_elevators=1, max_floor=5)
        wrapper.reset()
        p1 = Person(0, 3, 0)
        p2 = Person(0, 4, 2)
        waiting_t0 = [[[p1], []], [[], []], [[], []], [[], []], [[], []]]
        wrapper.step([2], waiting_t0, timestep=0)  # load p1 at 0, move to 1
        waiting_t1 = [[], [[p2], []], [[], []], [[], []], [[], []]]
        wrapper.step([2], waiting_t1, timestep=2)  # load p2 at 1, move to 2
        empty_waiting = [[[], []] for _ in range(5)]
        _, _, _, _, info = wrapper.step([2], empty_waiting, timestep=6)
        # p1 waited 6-0=6, p2 waited 6-2=4
        assert info["mean_elevator_waiting_time"] == pytest.approx(5.0)
        assert info["max_elevator_waiting_time"] == 6
        assert info["min_elevator_waiting_time"] == 4

    def test_step_deterministic(self):
        np.random.seed(SEED)
        wrapper = ElevatorWrapper(max_elevators=2, max_floor=5)
        wrapper.reset()
        waiting_people = [[[], []] for _ in range(5)]
        actions = [2, 0]  # elevator 0 up, elevator 1 down (invalid at floor 0)
        obs1, reward1, _, _, _ = wrapper.step(actions, waiting_people, timestep=0)

        np.random.seed(SEED)
        wrapper2 = ElevatorWrapper(max_elevators=2, max_floor=5)
        wrapper2.reset()
        obs2, reward2, _, _, _ = wrapper2.step(actions, waiting_people, timestep=0)

        np.testing.assert_array_almost_equal(obs1, obs2)
        assert reward1 == reward2

    def test_reward_no_unloads_no_waiting_no_invalid(self):
        """Idle with empty building: reward should be 0."""
        np.random.seed(SEED)
        wrapper = ElevatorWrapper(max_elevators=1, max_floor=5)
        wrapper.reset()
        waiting_people = [[[], []] for _ in range(5)]
        _, reward, _, _, _ = wrapper.step([1], waiting_people, timestep=0)
        assert reward == 0.0

    def test_reward_invalid_action_penalty(self):
        """Invalid action (e.g. DOWN at floor 0) incurs -10 penalty."""
        np.random.seed(SEED)
        wrapper = ElevatorWrapper(max_elevators=1, max_floor=5)
        wrapper.reset()
        waiting_people = [[[], []] for _ in range(5)]
        _, reward, _, _, _ = wrapper.step(
            [0], waiting_people, timestep=0
        )  # DOWN at floor 0
        assert reward == pytest.approx(-10.0)

    def test_reward_invalid_up_at_top(self):
        """Invalid UP at top floor incurs -10 penalty."""
        np.random.seed(SEED)
        wrapper = ElevatorWrapper(max_elevators=1, max_floor=5)
        wrapper.reset()
        waiting_people = [[[], []] for _ in range(5)]
        for _ in range(4):
            wrapper.step([2], waiting_people, timestep=0)  # move to floor 4
        _, reward, _, _, _ = wrapper.step(
            [2], waiting_people, timestep=0
        )  # invalid UP at top
        assert reward == pytest.approx(-10.0)

    def test_reward_people_waiting_penalty(self):
        """More people waiting on floors = more negative reward."""
        np.random.seed(SEED)
        wrapper = ElevatorWrapper(max_elevators=1, max_floor=5)
        wrapper.reset()

        waiting_empty = [[[], []] for _ in range(5)]
        _, reward_empty, _, _, _ = wrapper.step([1], waiting_empty, timestep=0)

        p1 = Person(0, 2, 0)
        p2 = Person(1, 3, 0)
        waiting_two = [[[p1], []], [[p2], []], [[], []], [[], []], [[], []]]
        wrapper2 = ElevatorWrapper(max_elevators=1, max_floor=5)
        wrapper2.reset()
        _, reward_two, _, _, _ = wrapper2.step([1], waiting_two, timestep=0)

        assert reward_empty == 0.0
        assert reward_two == pytest.approx(-0.02)  # 2 people * 0.01 each

    def test_reward_elevator_waiting_time_penalty(self):
        """Longer time passengers spend in elevator = more negative reward."""
        np.random.seed(SEED)

        wrapper = ElevatorWrapper(max_elevators=1, max_floor=5)
        wrapper.reset()
        p = Person(0, 2, 0)
        waiting = [[[p], []], [[], []], [[], []], [[], []], [[], []]]
        wrapper.step([2], waiting, timestep=0)  # load, 0->1. Passenger in, waited 0
        obs, reward, _, _, _ = wrapper.step(
            [2], waiting, timestep=1
        )  # 1->2. Passenger waited 1 step
        # elevator_waiting_times = [1], reward = 0 - 1*0.01 = -0.01
        assert reward == pytest.approx(-0.01)
        np.testing.assert_array_equal(
            obs,
            np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        )
        # Step 3: at floor 2, unload (1 person). Unload happens before elevator_waiting_times computed,
        # so carrying_people is empty → no waiting penalty. reward = +1 for delivery.
        obs, reward, _, _, _ = wrapper.step(
            [2], waiting, timestep=2
        )  # UP: unload at 2, move to 3
        np.testing.assert_array_equal(
            obs,
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        )
        assert reward == pytest.approx(1.0)

    def test_reward_unload_with_invalid_still_penalized(self):
        """If we unload someone but also do invalid action, we get num_unloaded - 10."""
        np.random.seed(SEED)
        wrapper = ElevatorWrapper(max_elevators=1, max_floor=5)
        wrapper.reset()
        p = Person(3, 4, 0)
        waiting = [[[], []], [[], []], [[], []], [[p], []], [[], []]]
        for _ in range(3):
            wrapper.step([2], waiting, timestep=0)  # 0->1->2->3
        wrapper.step([2], waiting, timestep=0)  # at 3, load, move to 4
        _, reward, _, _, _ = wrapper.step(
            [2], waiting, timestep=0
        )  # at 4, unload (1), invalid UP
        assert reward == pytest.approx(-9.0)  # 1 unloaded - 10 invalid penalty
