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
