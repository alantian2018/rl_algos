import gymnasium as gym
import numpy as np
from collections import deque
import itertools
import cv2


class SnakeEnv(gym.Env):
    def __init__(self, grid_height, grid_width, max_steps, render_mode=None):
        super().__init__()
        #  U, D, L, R
        self.action_space = gym.spaces.Discrete(4)
        # encode as 3 channels snake, snake body, food
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(grid_height, grid_width, 3)
        )
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.ALL_GRID_POSITIONS = set(
            itertools.product(range(self.grid_height), range(self.grid_width))
        )

    def _get_valid_food_position(self):
        possible = self.ALL_GRID_POSITIONS - self.snake_positions
        idx = np.random.choice(range(len(possible)))

        return list(possible)[idx]

    def reset(self):
        super().reset()
        self.env_step = 0
        self.grid = np.zeros((self.grid_height, self.grid_width, 3))

        # queue of our snake. we can take the tail and put it to the head
        self.snake = deque()
        # set for quick lookup of snake positions
        self.snake_positions = set()

        snake_position = (
            np.random.randint(0, self.grid_height),
            np.random.randint(0, self.grid_width),
        )
        self.snake_positions.add(snake_position)
        self.snake.appendleft(snake_position)
        self.grid[snake_position + (0,)] = 1

        # choose a random food position not equal to the snake

        self.food_position = self._get_valid_food_position()
        self.grid[self.food_position + (1,)] = 1

        return self.grid, {
            "snake_positions": self.snake_positions,
            "food_position": self.food_position,
        }

    def step(self, action):
        # action is [0,1,2,3] -> [U, D, L, R]
        has_eaten = False
        x, y = self.snake[0]
        xx, yy = x, y

        if action == 0:
            xx -= 1
        elif action == 1:
            xx += 1
        elif action == 2:
            yy -= 1
        elif action == 3:
            yy += 1

        new_position = (xx, yy)
        info = {"x": x, "y": y, "new_x": xx, "new_y": yy}

        # Check if eating food
        if new_position == self.food_position:
            has_eaten = True

        # Hit wall
        if xx < 0 or xx >= self.grid_height or yy < 0 or yy >= self.grid_width:
            return self.grid, -1, True, False, info
        # Hit self (but not the tail, which will move away if not eating)
        if new_position in self.snake_positions:
            if new_position == self.snake[-1] and not has_eaten:
                pass
            else:
                return self.grid, -1, True, False, info

        # Update snake position
        self.grid[new_position + (0,)] = 1  # new head
        self.grid[x, y, 0] = 0  # old head becomes body
        self.grid[x, y, 2] = 1
        self.snake_positions.add(new_position)
        self.snake.appendleft(new_position)

        # Shrink the snake from the tail (unless eating)
        if not has_eaten:
            tail = self.snake.pop()
            self.grid[tail + (2,)] = 0
            if tail != new_position:
                self.snake_positions.remove(tail)
        else:
            # Eating: spawn new food
            self.grid[self.food_position + (1,)] = 0
            self.food_position = self._get_valid_food_position()
            self.grid[self.food_position + (1,)] = 1

        self.env_step += 1

        return (
            self.grid,
            1 if has_eaten else -0.001,
            False,
            self.env_step >= self.max_steps,
            info,
        )

    def render(self, mode="rgb_array"):

        if mode is None:
            mode = self.render_mode
        if mode is None:
            raise ValueError("Invalid mode: None")

        SNAKE_HEAD_COLOR = np.array([0, 0, 255], dtype=np.uint8)  # blue
        SNAKE_BODY_COLOR = np.array([173, 216, 230], dtype=np.uint8)  # lightblue
        FOOD_COLOR = np.array([255, 0, 0], dtype=np.uint8)  # red
        EMPTY_COLOR = np.array([255, 255, 255], dtype=np.uint8)  # white
        img = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
        img[:] = EMPTY_COLOR
        # Draw food (channel 1)
        food_mask = self.grid[:, :, 1] == 1
        img[food_mask] = FOOD_COLOR

        body_mask = self.grid[:, :, 2] == 1
        img[body_mask] = SNAKE_BODY_COLOR

        head_mask = self.grid[:, :, 0] == 1
        img[head_mask] = SNAKE_HEAD_COLOR

        # Scale up for visibility (optional, makes video clearer)
        scale = 20
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

        # Add score text (snake length)
        score_text = f"Length: {len(self.snake)}"
        cv2.putText(
            img, score_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )

        if mode == "rgb_array":
            return img
        elif mode == "human":
            # cv2 uses BGR, so convert from RGB
            cv2.imshow("Snake", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def close(self):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    env = SnakeEnv(10, 10, 100)
    grid, info = env.reset()
    print(info)

    actions = {"U": 0, "D": 1, "L": 2, "R": 3}
    env.render(mode="human")

    for i in range(100):
        action = None

        while action not in actions:
            action = input("Enter an action: ").upper()

        obs, reward, terminated, truncated, info = env.step(actions[action])
        if terminated or truncated:
            env.reset()
        env.render(mode="human")
        print(
            f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}"
        )

    env.close()
