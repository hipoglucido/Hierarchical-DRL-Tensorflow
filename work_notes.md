# Work Notes

## 2016-09-11

### Status
I've managed to get some results on the MDP environment, but they are not as good as the paper and have not been pushed to the repo yet. I had to take a hiatus recently to move my apartment and start a new semester at McGill.

### Plan of Attack
I'd like to build an automated system for testing network architectures and recording the cumulative regret over the course of 12000 episodes

1. Abstract h-DQN into separate file from `run.py`
2. Modify h-DQN to accept network architecture hyperparameters
3. Create `search_architectures.py`: randomly select network architecture, test it, and append results to data file
4. Create `plot_architectures.Rmd`: knitr document for evaluating the network architectures that have been tested
5. Settle on satisfactory network architecture for h-DQN default values
6. Use default values to update `README.md` with results and discussion
7. Move on to Montezuma's revenge

### Notes

#### Abstract h-DQN into separate file from `run.py`

- This was easy to do. Only took a few minutes.

#### Modify h-DQN to accept network architecture hyperparameters

- This also went smoothly.

#### Create `search_architectures.py`: randomly select network architecture, test it, and append results to data file

- Was tedious but easy to create the ability to randomly select network architectures
- Didn't think about the fact that I'm not currently recording cumulative regret
  - Needed to implement this feature as part of this task
- Also realized there were a bunch of other parameters worth searching through
  - Pulled these out so that `search_architectures.py` can pass them in as params
- There seems to be some sort of bug that occurs when some architectures are randomly selected
  - Finished debugging/fixing this issue. Needed to add a `np.expand_dims()` call whenever the number of experience samples was set to 1
- Now outputing results to data file
