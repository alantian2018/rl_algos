"""
Tests for ElevatorEnv (the full gym environment wrapper).
Run from repo root: python -m pytest env/elevator/v1/testing/ -v
"""

import pytest
import numpy as np

from ..src.elevator_env import ElevatorEnv
from ..src.building import Person

SEED = 42
NUM_FLOORS = 5
NUM_ELEVATORS = 1
IDLE, DOWN, UP = 1, 0, 2


@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(SEED)


def obs_size(num_floors, num_elevators):
    floor_obs = num_floors * 2
    per_elevator = num_floors * 2 + 3  # floor one-hot + target one-hot + action one-hot
    return floor_obs + num_elevators * per_elevator


def make_env(num_elevators=NUM_ELEVATORS, num_floors=NUM_FLOORS, **kwargs):
    return ElevatorEnv(
        num_elevators=num_elevators,
        num_floors=num_floors,
        max_steps=kwargs.pop("max_steps", 100),
        spawn_rate=kwargs.pop("spawn_rate", 0.0),  # no random spawns by default
        render_mode=kwargs.pop("render_mode", "rgb_array"),
        **kwargs,
    )


# ── Reset ────────────────────────────────────────────────────────────────────


class TestReset:
    def test_obs_shape(self):
        env = make_env()
        obs, _ = env.reset()
        assert obs.shape == (obs_size(NUM_FLOORS, NUM_ELEVATORS),)

    def test_step_counter_zeroed(self):
        env = make_env()
        env.step([IDLE])
        env.step([IDLE])
        env.reset()
        assert env.current_step == 0

    def test_elevators_at_ground_floor(self):
        env = make_env(num_elevators=2)
        env.reset()
        for elev in env.elevator_wrapper.elevators:
            assert elev.current_floor == 0

    def test_no_people_waiting(self):
        env = make_env()
        env.reset()
        _, people = env.building.get_building_state()
        assert np.all(people == 0)

    def test_no_people_in_elevators(self):
        env = make_env()
        env.reset()
        for elev in env.elevator_wrapper.elevators:
            assert len(elev.carrying_people) == 0

    def test_reset_clears_previous_state(self):
        env = make_env(spawn_rate=1.0)
        for _ in range(20):
            env.step([UP])
        env.reset()
        assert env.current_step == 0
        _, people = env.building.get_building_state()
        assert np.all(people == 0)
        for elev in env.elevator_wrapper.elevators:
            assert elev.current_floor == 0
            assert len(elev.carrying_people) == 0


# ── Step basics ──────────────────────────────────────────────────────────────


class TestStepBasics:
    def test_returns_five_values(self):
        env = make_env()
        result = env.step([IDLE])
        assert len(result) == 5
        obs, reward, done, truncated, info = result

    def test_obs_shape_after_step(self):
        env = make_env()
        obs, *_ = env.step([IDLE])
        assert obs.shape == (obs_size(NUM_FLOORS, NUM_ELEVATORS),)

    def test_step_increments_counter(self):
        env = make_env()
        env.step([IDLE])
        assert env.current_step == 1
        env.step([IDLE])
        assert env.current_step == 2

    def test_truncated_always_false(self):
        env = make_env()
        _, _, _, truncated, _ = env.step([IDLE])
        assert truncated is False

    def test_info_is_dict(self):
        env = make_env()
        _, _, _, _, info = env.step([IDLE])
        assert isinstance(info, dict)


# ── Done condition ───────────────────────────────────────────────────────────


class TestDoneCondition:
    def test_not_done_before_max_steps(self):
        env = make_env(max_steps=5)
        for _ in range(4):
            _, _, done, _, _ = env.step([IDLE])
            assert done is False

    def test_done_at_max_steps(self):
        env = make_env(max_steps=5)
        for _ in range(4):
            env.step([IDLE])
        _, _, done, _, _ = env.step([IDLE])
        assert done is True


# ── Elevator movement ────────────────────────────────────────────────────────


class TestElevatorMovement:
    def test_up_moves_elevator(self):
        env = make_env()
        env.step([UP])
        assert env.elevator_wrapper.elevators[0].current_floor == 1

    def test_down_moves_elevator(self):
        env = make_env()
        env.step([UP])
        env.step([UP])
        env.step([DOWN])
        assert env.elevator_wrapper.elevators[0].current_floor == 1

    def test_idle_stays(self):
        env = make_env()
        env.step([UP])
        env.step([IDLE])
        assert env.elevator_wrapper.elevators[0].current_floor == 1

    def test_full_trip_up(self):
        env = make_env()
        for _ in range(NUM_FLOORS - 1):
            env.step([UP])
        assert env.elevator_wrapper.elevators[0].current_floor == NUM_FLOORS - 1

    def test_obs_reflects_elevator_position(self):
        env = make_env()
        env.step([UP])
        env.step([UP])
        elev = env.elevator_wrapper.elevators[0]
        state = elev.get_state()
        floor_one_hot = state[:NUM_FLOORS]
        assert floor_one_hot[2] == 1
        assert sum(floor_one_hot) == 1


# ── Reward ───────────────────────────────────────────────────────────────────


class TestReward:
    def test_idle_empty_building_zero_reward(self):
        env = make_env()
        _, reward, *_ = env.step([IDLE])
        assert reward == 0.0

    def test_invalid_action_penalty(self):
        env = make_env()
        _, reward, *_ = env.step([DOWN])  # DOWN at floor 0
        assert reward == pytest.approx(-10.0)

    def test_invalid_up_at_top(self):
        env = make_env()
        for _ in range(NUM_FLOORS - 1):
            env.step([UP])
        _, reward, *_ = env.step([UP])
        assert reward == pytest.approx(-10.0)

    def test_delivery_positive_reward(self):
        env = make_env()
        # Manually place a person at floor 0 going to floor 1
        env.building.waiting_people[0][0].append(Person(0, 1, 0))
        env.building.number_people_waiting = 1
        env.step([UP])  # load person going up, move to floor 1
        _, reward, *_ = env.step([IDLE])  # unload at floor 1
        assert reward == pytest.approx(1.0)

    def test_waiting_people_penalty(self):
        env = make_env()
        env.building.waiting_people[2][0].append(Person(2, 4, 0))
        env.building.waiting_people[3][1].append(Person(3, 0, 0))
        env.building.number_people_waiting = 2
        _, reward, *_ = env.step([IDLE])
        assert reward == pytest.approx(-0.02)  # 2 people * 0.01


# ── People spawning ─────────────────────────────────────────────────────────


class TestSpawning:
    def test_no_spawn_when_rate_zero(self):
        env = make_env(spawn_rate=0.0)
        for _ in range(50):
            env.step([IDLE])
        _, people = env.building.get_building_state()
        assert np.all(people == 0)

    def test_spawn_with_high_rate(self):
        env = make_env(spawn_rate=1.0)
        env.step([IDLE])
        _, people = env.building.get_building_state()
        assert np.sum(people) > 0

    def test_spawn_respects_max_people(self):
        env = make_env(spawn_rate=1.0, max_people=3)
        for _ in range(50):
            env.step([IDLE])
        assert env.building.number_people_waiting <= 3


# ── Loading / unloading integration ─────────────────────────────────────────


class TestLoadUnload:
    def test_load_unload_single_passenger(self):
        env = make_env()
        p = Person(0, 2, 0)
        env.building.waiting_people[0][0].append(p)
        env.building.number_people_waiting = 1

        env.step([UP])  # load at floor 0, move to 1
        elev = env.elevator_wrapper.elevators[0]
        assert len(elev.carrying_people) == 1

        env.step([UP])  # move to 2
        _, reward, *_ = env.step([IDLE])  # unload at 2
        assert reward == pytest.approx(1.0)
        assert len(elev.carrying_people) == 0

    def test_building_people_count_decreases_on_unload(self):
        env = make_env()
        p = Person(0, 1, 0)
        env.building.waiting_people[0][0].append(p)
        env.building.number_people_waiting = 1

        env.step([UP])  # load + move to 1; unload happens next step
        env.step([IDLE])  # unload
        assert env.building.number_people_waiting == 0

    def test_load_going_down(self):
        env = make_env()
        # Move elevator to floor 3 first
        for _ in range(3):
            env.step([UP])

        p = Person(3, 0, 0)
        env.building.waiting_people[3][1].append(p)
        env.building.number_people_waiting = 1

        env.step([DOWN])  # load person going down at floor 3, move to 2
        elev = env.elevator_wrapper.elevators[0]
        assert len(elev.carrying_people) == 1
        assert elev.carrying_people[0].target_floor == 0

    def test_multiple_pickups_and_deliveries(self):
        env = make_env()
        p1 = Person(0, 1, 0)
        p2 = Person(0, 3, 0)
        env.building.waiting_people[0][0].extend([p1, p2])
        env.building.number_people_waiting = 2

        env.step([UP])  # load both, move to 1
        elev = env.elevator_wrapper.elevators[0]
        assert len(elev.carrying_people) == 2

        _, reward, *_ = env.step([UP])  # unload p1 at 1, move to 2
        # +1 for delivery, -0.01 for p2 still riding (waited 1 step)
        assert reward == pytest.approx(0.99)
        assert len(elev.carrying_people) == 1

        env.step([UP])  # move to 3
        _, reward, *_ = env.step([IDLE])  # unload p2 at 3
        assert reward == pytest.approx(1.0)
        assert len(elev.carrying_people) == 0

    def test_idle_does_not_load(self):
        env = make_env()
        p = Person(0, 2, 0)
        env.building.waiting_people[0][0].append(p)
        env.building.number_people_waiting = 1

        env.step([IDLE])
        elev = env.elevator_wrapper.elevators[0]
        assert len(elev.carrying_people) == 0
        assert len(env.building.waiting_people[0][0]) == 1


# ── Multi-elevator ──────────────────────────────────────────────────────────


class TestMultiElevator:
    def test_obs_shape(self):
        env = make_env(num_elevators=3)
        obs, _ = env.reset()
        assert obs.shape == (obs_size(NUM_FLOORS, 3),)

    def test_independent_movement(self):
        env = make_env(num_elevators=2)
        env.step([UP, IDLE])
        elevs = env.elevator_wrapper.elevators
        assert elevs[0].current_floor == 1
        assert elevs[1].current_floor == 0

    def test_both_can_deliver(self):
        env = make_env(num_elevators=2)
        # Elevator 0 picks up from floor 0, elevator 1 stays idle
        p1 = Person(0, 1, 0)
        env.building.waiting_people[0][0].append(p1)
        env.building.number_people_waiting = 1
        env.step([UP, IDLE])  # elev0 loads p1, moves to 1; elev1 stays at 0
        _, reward, *_ = env.step([IDLE, IDLE])  # elev0 unloads at 1
        assert reward == pytest.approx(1.0)

        # Now elevator 1 picks up a new person from floor 0
        p2 = Person(0, 1, 0)
        env.building.waiting_people[0][0].append(p2)
        env.building.number_people_waiting = 1
        env.step([IDLE, UP])  # elev1 loads p2, moves to 1
        _, reward, *_ = env.step([IDLE, IDLE])  # elev1 unloads at 1
        assert reward == pytest.approx(1.0)


# ── Render ───────────────────────────────────────────────────────────────────


class TestRender:
    def test_rgb_array_returns_image(self):
        env = make_env(render_mode="rgb_array")
        img = env.render()
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.dtype == np.uint8

    def test_rgb_array_nonzero_dimensions(self):
        env = make_env(render_mode="rgb_array")
        img = env.render()
        assert img.shape[0] > 0
        assert img.shape[1] > 0

    def test_render_after_steps(self):
        env = make_env(render_mode="rgb_array")
        env.step([UP])
        env.step([UP])
        img = env.render()
        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 3

    def test_render_no_return_in_human_mode(self):
        """human mode should not return an image (it displays via cv2)."""
        env = make_env(render_mode="rgb_array")
        result = env.render(render_mode="rgb_array")
        assert result is not None


# ── Obs consistency ──────────────────────────────────────────────────────────


class TestObsConsistency:
    def test_obs_dtype_float(self):
        env = make_env()
        obs, _ = env.reset()
        assert obs.dtype in (np.float32, np.float64)

    def test_obs_shape_stable_across_steps(self):
        env = make_env(spawn_rate=0.5)
        expected = obs_size(NUM_FLOORS, NUM_ELEVATORS)
        for _ in range(20):
            obs, *_ = env.step([UP])
            assert obs.shape == (expected,)

    def test_reset_then_step_same_shape(self):
        env = make_env()
        reset_obs, _ = env.reset()
        step_obs, *_ = env.step([IDLE])
        assert reset_obs.shape == step_obs.shape
