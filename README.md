# RL_mountaincar
User friendly code to see Reinforcement Learning algorithm at work in a Mountain Car environment.
Mountain Car environment is based on the Example 9.2 from the book "Reinforcement Learning:an introduction" by Sutton and Barto.
It is an example of a RL environment with continuous state space where approximating the Q-value function is needed.


## How to start?
To start a simulation simply run
```
python mountain_car.py
```
which runs a pre-trained model on a randomly chosen initial states.

### What is going on?
In the top we sketch the environment of a hill with two barriers and the dot representing the car-agent. The task is to reach the top of the rightmost hill given any initial position and velocity.
Car-agent can do forward, backward or zero acceleration; its motion is in turn based on a simple physics model.
The difficulty is that the car-agent does not have enough power to go over the hill by just going forward.
Instead, in order to escape the valley and complete the task, the agent must learn a sequence of momentum storing phases.

Bottom plot is a phase space diagram with position (x-axis) and velocity (y-axis) of the car-agent.
Dashed grid is the approximation grid used to evaluate the Q function.  

## Options
All available options are found through
```
python mountain_car.py --help
```

## Algorithm

SARSA algorithm with a linear approximation of the Q function due to continuous state space.

## Examples
In the **/examples** directory working examples are provided.

- *back-and-forth*

A simulation shows how a pre-trained model work when the starting state is most unfortunate - the car-agent is at the bottom of the valley with no initial velocity.


```
./back-and-forth.sh
```
