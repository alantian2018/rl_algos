1. Finish environment (stepping, rendering, etc.)
    - Would be nice to demo an MVP with random actions and rendering + verifying env correctness
    - CLI type of rendering 
    
  ```
    +======================================================+
    |    floor n  |   |     Elevator 1    |                |
    |  people = 3 v   |  num_people = 2   |                |
    +======================================================+
    |   floor n-1 | ^ |                   |                |
    |  people = 0 v | |                   |                |
    +======================================================+
                               ...                         
    +======================================================+
    |    floor 2  |   |                   |                |
    |  people = 3 v   |                   |                |
    +======================================================+
    |   floor  1    ^ |                   |   Elevator 2   |
    |  people = 3   | |                   | num_people = 1 |
    +======================================================+
  ```

3. Try different loss functions
4. Testing. Use `python -m pytest env/elevator/v1/testing/ -v` to run all tests in the env
