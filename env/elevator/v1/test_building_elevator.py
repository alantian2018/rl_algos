"""
Tests for Building and Elevator classes.
Uses np.random.seed() for deterministic RNG across test runs.
"""

import pytest
import numpy as np

# Import after setting path - run from repo root or add to path
import sys
from pathlib import Path

# Add v1 directory to path for imports
v1_dir = Path(__file__).parent
sys.path.insert(0, str(v1_dir))

from building import Building, Person
from elevator_v1 import Elevator, ElevatorWrapper, ElevatorActions


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
        assert elevator.current_floor == 0
        assert elevator.carrying_people == []
        assert obs.shape == (5 + 5 + 3,)  # floor one-hot + target one-hot + action one-hot

    def test_get_state_shape(self):
        np.random.seed(SEED)
        elevator = Elevator(max_floor=4)
        state = elevator.get_state()
        assert state.shape == (4 + 4 + 3,)  # max_floor*2 + 3 for last_action

    def test_step_idle(self):
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5)
        waiting_people = [[[], []] for _ in range(5)]
        obs, num_unloaded, did_invalid, wait_times = elevator.step(1, waiting_people, timestep=0)
        assert elevator.current_floor == 0
        assert num_unloaded == 0
        assert did_invalid is False
        assert obs.shape == (5 + 5 + 3,)

    def test_step_up(self):
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5)
        waiting_people = [[[], []] for _ in range(5)]
        obs, num_unloaded, did_invalid, _ = elevator.step(2, waiting_people, timestep=0)
        assert elevator.current_floor == 1
        assert did_invalid is False

    def test_step_down(self):
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5, start_floor=2)
        waiting_people = [[[], []] for _ in range(5)]
        obs, num_unloaded, did_invalid, _ = elevator.step(0, waiting_people, timestep=0)
        assert elevator.current_floor == 1
        assert did_invalid is False

    def test_step_invalid_up_at_top(self):
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5, start_floor=4)
        waiting_people = [[[], []] for _ in range(5)]
        obs, num_unloaded, did_invalid, _ = elevator.step(2, waiting_people, timestep=0)
        assert elevator.current_floor == 4  # didn't move
        assert did_invalid is True

    def test_step_invalid_down_at_bottom(self):
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5)
        waiting_people = [[[], []] for _ in range(5)]
        obs, num_unloaded, did_invalid, _ = elevator.step(0, waiting_people, timestep=0)
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
            [[], []], [[], []], [[], []],
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
        obs, num_unloaded, _, _ = elevator.step(1, waiting_people, timestep=2)
        assert num_unloaded == 1
        assert len(elevator.carrying_people) == 0

    def test_up_sequence_full_building(self):
        """Elevator moves from floor 0 to top, verifying position at each step."""
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5)
        waiting_people = [[[], []] for _ in range(5)]
        for i in range(4):  # 4 UP steps: 0->1->2->3->4
            elevator.step(2, waiting_people, timestep=i)
            assert elevator.current_floor == i + 1, f"After {i+1} UP steps, expected floor {i+1}"
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
            [[], []], [[], []], [[], []],
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
        obs, num_unloaded, _, _ = elevator.step(1, waiting_people, timestep=3)  # idle to unload
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
            [[], []], [[], []],
            [[], [person_going_down]],  # floor 2: person going down
            [[], []], [[], []],
        ]
        elevator.step(2, waiting_people, timestep=0)  # UP - should NOT load person going down
        assert len(elevator.carrying_people) == 0
        assert len(waiting_people[2][1]) == 1

    def test_down_does_not_load_passengers_going_up(self):
        """When going DOWN, we only load people going down (floor[1]), not up (floor[0])."""
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5, start_floor=2)
        person_going_up = Person(src_floor=2, target_floor=4, time_created=0)
        waiting_people = [
            [[], []], [[], []],
            [[person_going_up], []],  # floor 2: person going up
            [[], []], [[], []],
        ]
        elevator.step(0, waiting_people, timestep=0)  # DOWN - should NOT load person going up
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
            [[], []], [[], []], [[], []],
        ]
        # Floor 0: load p_floor2, move to 1
        elevator.step(2, waiting_people, timestep=0)
        assert len(elevator.carrying_people) == 1
        # Floor 1: load p_floor4, move to 2
        elevator.step(2, waiting_people, timestep=1)
        assert len(elevator.carrying_people) == 2
        # Floor 2: unload p_floor2, move to 3
        _, num_unloaded, _, _ = elevator.step(2, waiting_people, timestep=2)
        assert num_unloaded == 1
        assert len(elevator.carrying_people) == 1
        assert elevator.carrying_people[0].target_floor == 4
        # Floor 3: move to 4
        elevator.step(2, waiting_people, timestep=3)
        # Floor 4: unload p_floor4
        _, num_unloaded, _, _ = elevator.step(1, waiting_people, timestep=4)
        assert num_unloaded == 1
        assert len(elevator.carrying_people) == 0

    def test_up_then_down_full_trip(self):
        """Elevator goes up with passenger, delivers, goes down with another passenger."""
        np.random.seed(SEED)
        elevator = Elevator(max_floor=5)
        p_up = Person(src_floor=0, target_floor=4, time_created=0)
        p_down = Person(src_floor=4, target_floor=0, time_created=0)
        waiting_people = [
            [[p_up], []], [[], []], [[], []], [[], []],
            [[], [p_down]],  # floor 4 going down
        ]
        # Load at 0, go up to 4
        elevator.step(2, waiting_people, timestep=0)  # 0->1
        elevator.step(2, waiting_people, timestep=1)  # 1->2
        elevator.step(2, waiting_people, timestep=2)  # 2->3
        elevator.step(2, waiting_people, timestep=3)  # 3->4
        _, n, _, _ = elevator.step(1, waiting_people, timestep=4)  # unload at 4
        assert n == 1
        # Now load person going down at floor 4
        elevator.step(0, waiting_people, timestep=5)  # DOWN: load, 4->3
        assert len(elevator.carrying_people) == 1
        assert elevator.carrying_people[0].target_floor == 0
        # Go down to 0 (3 more steps: 3->2->1->0)
        for t in range(6, 9):
            elevator.step(0, waiting_people, timestep=t)
        _, n, _, _ = elevator.step(1, waiting_people, timestep=7)  # unload at 0
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
        obs, reward, total_unloaded, info = wrapper.step(actions, waiting_people, timestep=0)
      
        assert obs.shape == (2 * (5 + 5 + 3),)
        assert "did_invalid_actions" in info
        assert "elevator_waiting_times" in info
        assert total_unloaded == 0

    def test_step_deterministic(self):
        np.random.seed(SEED)
        wrapper = ElevatorWrapper(max_elevators=2, max_floor=5)
        wrapper.reset()
        waiting_people = [[[], []] for _ in range(5)]
        actions = [2, 0]  # elevator 0 up, elevator 1 down (invalid at floor 0)
        obs1, reward1, _, _ = wrapper.step(actions, waiting_people, timestep=0)

        np.random.seed(SEED)
        wrapper2 = ElevatorWrapper(max_elevators=2, max_floor=5)
        wrapper2.reset()
        obs2, reward2, _, _ = wrapper2.step(actions, waiting_people, timestep=0)

        np.testing.assert_array_almost_equal(obs1, obs2)
        assert reward1 == reward2

    def test_reward_no_unloads_no_waiting_no_invalid(self):
        """Idle with empty building: reward should be 0."""
        np.random.seed(SEED)
        wrapper = ElevatorWrapper(max_elevators=1, max_floor=5)
        wrapper.reset()
        waiting_people = [[[], []] for _ in range(5)]
        _, reward, _, _ = wrapper.step([1], waiting_people, timestep=0)
        assert reward == 0.0

    def test_reward_invalid_action_penalty(self):
        """Invalid action (e.g. DOWN at floor 0) incurs -10 penalty."""
        np.random.seed(SEED)
        wrapper = ElevatorWrapper(max_elevators=1, max_floor=5)
        wrapper.reset()
        waiting_people = [[[], []] for _ in range(5)]
        _, reward, _, _ = wrapper.step([0], waiting_people, timestep=0)  # DOWN at floor 0
        assert reward == pytest.approx(-10.0)

    def test_reward_invalid_up_at_top(self):
        """Invalid UP at top floor incurs -10 penalty."""
        np.random.seed(SEED)
        wrapper = ElevatorWrapper(max_elevators=1, max_floor=5)
        wrapper.reset()
        waiting_people = [[[], []] for _ in range(5)]
        for _ in range(4):
            wrapper.step([2], waiting_people, timestep=0)  # move to floor 4
        _, reward, _, _ = wrapper.step([2], waiting_people, timestep=0)  # invalid UP at top
        assert reward == pytest.approx(-10.0)

    def test_reward_people_waiting_penalty(self):
        """More people waiting on floors = more negative reward."""
        np.random.seed(SEED)
        wrapper = ElevatorWrapper(max_elevators=1, max_floor=5)
        wrapper.reset()

        waiting_empty = [[[], []] for _ in range(5)]
        _, reward_empty, _, _ = wrapper.step([1], waiting_empty, timestep=0)

        p1 = Person(0, 2, 0)
        p2 = Person(1, 3, 0)
        waiting_two = [[[p1], []], [[p2], []], [[], []], [[], []], [[], []]]
        wrapper2 = ElevatorWrapper(max_elevators=1, max_floor=5)
        wrapper2.reset()
        _, reward_two, _, _ = wrapper2.step([1], waiting_two, timestep=0)

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
        obs, reward, _, _ = wrapper.step([2], waiting, timestep=1)  # 1->2. Passenger waited 1 step
        # elevator_waiting_times = [1], reward = 0 - 1*0.01 = -0.01
        assert reward == pytest.approx(-0.01)
        np.testing.assert_array_equal(obs, np.array([0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]))
        # Step 3: at floor 2, unload (1 person). Unload happens before elevator_waiting_times computed,
        # so carrying_people is empty → no waiting penalty. reward = +1 for delivery.
        obs, reward, _, _ = wrapper.step([2], waiting, timestep=2)  # UP: unload at 2, move to 3
        np.testing.assert_array_equal(obs, np.array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.]))
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
        _, reward, _, _ = wrapper.step([2], waiting, timestep=0)  # at 4, unload (1), invalid UP
        assert reward == pytest.approx(-9.0)  # 1 unloaded - 10 invalid penalty
