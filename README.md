# h-DQN

Reproduction of "Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation" by Kulkarni et al. (2016) in Python: https://arxiv.org/abs/1604.06057

## Disclaimer

This is a work in progress. I haven't been able to replicate the results yet.

Also, I haven't started on Montezuma's revenge yet. I intend to do this eventually, but I'm not sure when. Pull requests are welcomed and encouraged!

Comments/criticisms/suggestions/etc welcome, as always.

## Progress

### MDP Environment
- Create MDP Environment **[Done]**
- Create a non-hierarchical actor-critic agent as a baseline **[Done]**
- Evaluate the non-hierachical actor-critic by plotting which states it visits **[Done]**
- Create a h-DQN agent **[Done]**
- Evaluate the h-DQN agent by plotting which states it visits **[In progress]**

### Montezuma's Revenge

TODO (This might be a while. Pull requests welcome.)

### Work Notes

I've started [keeping track of experiments and work](https://github.com/EthanMacdonald/h-DQN/blob/master/work_notes.md) notes for posterity.

## Results

### Stochastic MDP Environment

#### Simple Actor-Critic

The simple actor-critic agent defined in `actor_critic.py` doesn't learn to visit state 6 very much at all. This is exactly what we might expect from this sort of agent in this sort of environment. This agent is simply a baseline for comparison.

![State visits](https://github.com/EthanMacdonald/h-DQN/blob/master/fig/mdp-actor-critic-visits.png)

Fig 1. Each subplot shows the number of times the actor-critic agent visited a given state (averaged over 1000 episodes).

To reproduce the figure above run `python actor_critic.py` from the command line.

#### h-DQN

The h-DQN agent is located in `./agent/hDQN.py`. So far I haven't been able to produce comparable results to the paper, but I have been getting better than baseline.

There are a few reasons this could be happening. I might have a critical bug(s) somewhere in my code or I might not have the right network architecture. I've added the file `search_architectures.py` which essentially chooses a random architecture, runs it for 12000 episodes, and prints the results to a CSV file. My next step will be creating a Knitr document for analyzing the output.

One particularly odd pattern I'm noticing is a tendency for the agent to seemingly converge to selecting the correct (or *almost* correct) subgoal only to watch performance taper off in later episodes. It's almost like the poor little guy gets tired and loses hope. :'(

## Requirements

- numpy
- tensorflow
- keras
- h5py
- matplotlib
