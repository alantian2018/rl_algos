## elevatorz

1. physics? probably too hard for prototype but would be interested in implementing this in a bit
2. implement some form of ddqn? would most likely perform better than ppo.
3. really wanna benchmark classic elevators (ie up/down button at source floor, choose target floor inside elevator) vs choose target floor from source floor too see how much more efficient it is

## marl brainstorming
1. naively just encode all other options as static/part of the env. I think this could work => train one shared network. This could work for desynced action execution (ie only one elevator action is stepped at a time)

2. if we sync actions per observation I wanna try out CTDE cuz it looks interesting
---
# Plans for V1:
Action Space for v1: up, down, idle
Observation space: People States, elevator floors (store target floors of current ppl inside)

reward = mse on people waiting time.

## how to design classes?
- Building (both number, source floor, and signal it gives to the building (up/down or target floor))
    - bernoulli for now, poisson later(?)
    - Observation -> floor, call elevator signal
- elevator (should be independent of people states) 
    - Step -> up/down/idle for now...
    -  Observation -> Target floors of people in elevator, current elevator floor.
    
- elevator wrapper class methods
    - mask steps out for nonconcurrent actions later

- Elevator env needs to tie everything together
    - step (for all elevators)
    - observation (people obs + elevator obs)
    - reset -> floor 0 idle for all elevs

## step execution flow
1. apply action
    - load people in if they are going the right way (more realism to come). For example, if action is UP  and elevator at floor 4 -> load UP people on floor 4. Light up their target floors.

