# h-DQN

Reproduction of "Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation" by Kulkarni et al. (2016) in Python: https://arxiv.org/abs/1604.06057

## Disclaimer

This is totally a work in progress. As of this writing (Aug 21, 2016) I only started working on this yesterday evening. :)

Also, I haven't started on Montezuma's revenge yet. But I do intend to.

Comments/criticisms/suggestions/pull requests welcome, as always.

## Progress

### MDP Environment
- Create MDP Environment **[Done]**
- Create a non-hierarchical actor-critic agent as a baseline **[Done]**
- Evaluate the non-hierachical actor-critic by plotting which states it visits **[Done]**
- Create a h-DQN agent **[Done]**
- Evaluate the h-DQN agent by plotting which states it visits **[In progress]**

### Montezuma's Revenge

TODO

## Results

### Stochastic MDP Environment

#### Simple Actor-Critic

The simple actor-critic methods defined in `actor_critic.py` doesn't learn to visit state 6 very much at all. This is exactly what we might expect from this sort of agent in this sort of environment. This agent is simply a baseline for comparison.

![State visits](https://github.com/EthanMacdonald/h-DQN/blob/master/fig/mdp-actor-critic-visits.png)

Fig 1. Each subplot shows the number of times the actor-critic agent visited a given state (averaged over 1000 episodes).

To reproduce the figure above run `python actor_critic.py` from the command line.

## Requirements

- numpy
- tensorflow
- keras
- h5py
- matplotlib
