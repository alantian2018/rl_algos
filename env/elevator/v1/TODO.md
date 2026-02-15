1. Finish environment (stepping, rendering, etc.)
    - Would be nice to demo an MVP with random actions and rendering + verifying env correctness
    - CLI type of rendering 
    
  ```
    |======================================================|
    |    floor n  |   |     Elevator 1    |                |
    |  people = 3 v   |  num_people = 2   |                |
    |=======================================================
    |   floor n-1 | ^ |                   |                |
    |  people = 0 v | |                   |                |
    |======================================================|
                               ...                         
    |======================================================|
    |    floor 2  |   |                   |                |
    |  people = 3 v   |                   |                |
    |=======================================================
    |   floor  1    ^ |                   |   Elevator 2   |
    |  people = 3   | |                   | num_people = 1 |
    |======================================================|
  ```
2. Implment MultiDiscrete version of PPO
    - We can test with 1 elevator first to skirt this, but will be necessary when we have >2 elevator envs. Action space is equivalent to MultiDiscrete in SB3, e.g. 
    `[action] * num_elevators`
