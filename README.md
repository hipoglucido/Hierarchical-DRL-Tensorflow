# h-DQN
Reproduction of https://arxiv.org/pdf/1604.06057v2.pdf

# Warning

This is totally a work in progress. As of this writing (Aug 21, 2016) I only started working on this yesterday. :)

Also, I haven't started on Montezuma's revenge yet. But I do intend to.

Comments/criticisms/suggestions/pull requests welcome, as always.

# Results

## Stochastic MDP Environment

### Simple Actor-Critic

The simple actor-critic methods defined in `actor_critic.py` doesn't learn to visit state 6 very much at all.

![State visits](https://github.com/EthanMacdonald/h-DQN/blob/master/fig/mdp-actor-critic-visits.png)

Fig 1. Each subplot shows the number of times the actor-critic agent visited a given state (averaged over 1000 episodes).

To reproduce the figure above run `python actor_critic.py` from the command line

## Requirements

- numpy
- tensorflow
- keras
- h5py
- matplotlib
