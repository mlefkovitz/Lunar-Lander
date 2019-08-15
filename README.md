# Lunar Lander

## Introduction

In this project, I develop a reinforcement learning algorithm to learn to land a rocket on a landing pad
by controlling rocket engine actions. OpenAI Gym’s Lunar Lander is an environment that takes in one of 4
discrete actions at each time step returns a state in an 8-dimensional continuous state space along with a
reward.

I designed a Policy Gradient algorithm to solve this problem. The goal of the policy gradient method is to learn neural network weights that describe the best action for the rocket to take in each state. The neural network takes in the information about the current state (where the rocket is in space), processes it through a single hidden layer, then outputs the probability that the rocket should take each action. 

As the algorithm sees various states through repeated episodes, it maps rewards to the actions taken at each of those states and develops a policy that maximizes reward. On a regular basis, the weights of the neural network describing the policy are updated using the gradient between actual reward and expected reward (backpropogation algorithm).

The image to the right shows a successful landing using the policy gradient algorithm after over 3000 training episodes. You can see my project report below. 

[Original Project Requirments](/CS7642_Project2.pdf)

[Project Report](/Lunar%20Lander%20Project%20Report.pdf)

Successful Landing after Policy Gradient learning:

![Successful landing](PG%20animation%20-%20one%20good%20episode.gif)

## The Project
Python Files:
- Lunar Lander Heuristic.py
- Lunar Lander PG.py
- Lunar Lander PG Hyperparameters.py

### Lunar Lander Heuristic.py
- Uses the heuristic laid out in OpenAI's documentation
- Solves the problem in a single episode (no learning needed)

### Lunar Lander PG.py
- Implements numpy, gym, math, copy, and matplotlib.
- Seeded with seed = 100 (I like this number better than 1 or 0)
- Uses Policy Gradient algorithm originally from Karpathy's blog to solve the Lunar Lander problem
- Saves the best model before the maximum episodes and runs 100 trials against that model
- Plots all of the training episodes vs reward
- Plots all of the trials vs reward

### Lunar Lander PG Hyperparameters.py
- Same as above, but as a function that can be called with hyperparameters as inputs (and outputs to appropriately named files)
