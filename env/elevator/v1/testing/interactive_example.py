from os import terminal_size
from ..src.elevator_env import ElevatorEnv

env = ElevatorEnv(num_elevators=1, num_floors=10, max_steps=100, render_mode="human")

obs, _ = env.reset()
print(obs)
reward = 0
while True:
    env.render()
    print(f"Step {env.current_step}, Reward: {reward}")
    action = input("Enter action: ")
    if action == "r":
        # reset
        obs, _ = env.reset()

    if action == "q":
        break
    if action == "u":
        action = 2
    elif action == "d":
        action = 0
    elif action == "s":
        action = 1
    else:
        print("Invalid action")
        continue

    obs, reward, terminated, truncated, info = env.step([action] * env.num_elevators)

    if terminated or truncated:
        break
