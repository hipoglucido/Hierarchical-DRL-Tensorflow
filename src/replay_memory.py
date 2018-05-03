"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import os
import random

import numpy as np
import sys
sys.path.insert(0, '..')

from utils import save_npy, load_npy

class OldReplayMemory:
    def __init__(self, config, model_dir, screen_size):
        self.model_dir = model_dir     #TODO remove parameter
        self.memory_size = config.memory_size
        self.actions = np.empty(self.memory_size, dtype = np.uint8)
        self.rewards = np.empty(self.memory_size, dtype = np.float16)
        self.screens = np.empty((self.memory_size, screen_size), dtype = np.float16)
        self.terminals = np.empty(self.memory_size, dtype = np.bool)
        self.history_length = config.history_length
        self.dims = (screen_size,)
        self.batch_size = config.batch_size
        self.count = 0
        self.current = 0

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
        self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
    def is_full(self):    
        return self.count == self.memory_size
#    def add(self, screen, reward, action, terminal):
    def add(self, old_screen, action, reward, screen, terminal):
#        print("******4*********")
#        print(old_screen.reshape(5,5))
#        print(action, reward)
#        print(screen.reshape(5,5))
#        print(terminal)
        
        assert screen.shape == self.dims
        # NB! screen is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current, ...] = screen
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size
        

    def getState(self, index):
        assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.history_length - 1:
            # use faster slicing
            return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return self.screens[indexes, ...]

    def sample(self):
        # memory must include poststate, prestate and history
        
        #print(self.count, self.history_length)
        assert self.count > self.history_length
        
        # sample random indexes
        indexes = []
        while len(indexes) < self.batch_size:
            # find random index            
            while True:
                # sample one index (ignore states wraping over
                index = random.randint(self.history_length, self.count - 1)
                # if wraps over current pointer, then get new one
                if index >= self.current and index - self.history_length < self.current:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.terminals[(index - self.history_length):index].any():
                    continue
                # otherwise use this index
                break
            
            # NB! having index first is fastest in C-order matrices
            self.prestates[len(indexes), ...] = self.getState(index - 1)
            self.poststates[len(indexes), ...] = self.getState(index)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]
        #print("sample:",self.prestates, actions, rewards, self.poststates, terminals)
        return (self.prestates, actions, rewards, self.poststates, terminals), None, None, None, None

    def save(self):
        for idx, (name, array) in enumerate(
                zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
                        [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
            save_npy(array, os.path.join(self.model_dir, name))

    def load(self):
        for idx, (name, array) in enumerate(
                zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
                        [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
            array = load_npy(os.path.join(self.model_dir, name))
"""Replay Memory"""

import numpy as np
import random
from sum_tree import SumTree

class ReplayMemory:
    """Store and replay (sample) memories."""
#    def __init__(self,
#                max_size,
#                window_size,
#                input_shape):
    def __init__(self, config, model_dir, screen_size):
        max_size = config.memory_size
        self.batch_size = config.batch_size
        window_size = config.history_length
        input_shape = (screen_size,)
        """Setup memory.
        You should specify the maximum size o the memory. Once the
        memory fills up oldest values are removed.
        """
        self._max_size = max_size
        self._window_size = window_size
        self._WIDTH = input_shape[0]
        #self._HEIGHT = input_shape[1]
        self._memory = []
    @property
    def count(self): return len(self._memory)
        
    def is_full(self):
        return len(self._memory) == self._max_size
    def add(self, old_state, action, reward, new_state, is_terminal):
        """Add a list of samples to the replay memory."""
        num_sample = len(old_state)

        if len(self._memory) >= self._max_size:
            del(self._memory[0:num_sample])

#        for o_s, a, r, n_s, i_t in zip(old_state, action, reward, new_state, is_terminal):
#            self._memory.append((o_s, a, r, n_s, i_t))
#        print("*******7********")
#        print(old_state.reshape(5,5))
#        print(action, reward)
#        print(new_state.reshape(5,5))
#        print(is_terminal)
        self._memory.append((old_state.copy(), action, reward, new_state.copy(), is_terminal))

    def sample(self, indexes=None):
        """Return samples from the memory.
        Returns
        --------
        (old_state_list, action_list, reward_list, new_state_list, is_terminal_list, frequency_list)
        """
        samples = random.sample(self._memory, min(self.batch_size, len(self._memory)))
#        print(samples[0][0].reshape(5,5))
#        print('action',samples[0][1])
#        print('reward',samples[0][2])
#        print(samples[0][3].reshape(5,5))
#        print('t',samples[0][4])
        zipped = list(zip(*samples))

#        print(zipped[0][0].reshape(5,5))
#        print('action',zipped[1][0])
#        print('reward',zipped[2][0])
#        print(zipped[3][0].reshape(5,5))
#        print('t',zipped[4][0])
#        print("***********")
        zipped[0] = np.reshape(zipped[0], (-1, self._window_size, self._WIDTH))
        zipped[3] = np.reshape(zipped[3], (-1, self._window_size, self._WIDTH))
#        for z in zipped:
#            print(z)
        return zipped, None, None, None, None


class PriorityExperienceReplay:
    '''
    Almost copy from
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    '''
    def __init__(self, config, model_dir, screen_size):
        max_size = config.memory_size
        self.batch_size = config.batch_size
  
#    def __init__(self,
#                max_size,
#                window_size,
#                input_shape):
        self.tree = SumTree(max_size)
        self._max_size = max_size
        self._window_size = config.history_length
        self._WIDTH = screen_size#input_shape[0]
        #self._HEIGHT = 1#input_shape[1]
        self.e = 0.01
        self.a = 0.6
    @property
    def count(self): return self.tree.total_and_count()[1]
    def is_full(self): return self.count == self._max_size
    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, old_state, action, reward, new_state, is_terminal):
#        for o_s, a, r, n_s, i_t in zip(old_state, action, reward, new_state, is_terminal):
            # 0.5 is the maximum error
        error = abs(reward)#.5
        #print(reward, type(error))
        p = self._getPriority(error)
        self.tree.add(p, data=(old_state.copy(), action, reward, new_state.copy(), is_terminal)) 

    def sample(self, batch_size = None, indexes=None):
        data_batch = []
        idx_batch = []
        p_batch = []
        batch_size = self.batch_size
        segment = self.tree.total_and_count()[0] / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            data_batch.append(data)
            idx_batch.append(idx)
            p_batch.append(p)

        zipped = list(zip(*data_batch))
        zipped[0] = np.reshape(zipped[0], (-1, self._window_size, self._WIDTH))
        zipped[3] = np.reshape(zipped[3], (-1, self._window_size, self._WIDTH))
#        print(zipped[0][0].reshape(3,3))
#        print('action',zipped[1][0])
#        print('reward',zipped[2][0])
#        print(zipped[3][0].reshape(3,3))
#        print('t',zipped[4][0])
#        print("***********")

        sum_p, count = self.tree.total_and_count()
        return zipped, idx_batch, p_batch, sum_p, count

    def update(self, idx_list, error_list):
        for idx, error in zip(idx_list, error_list):
            p = self._getPriority(error)
            self.tree.update(idx, p)