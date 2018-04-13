# -*- coding: utf-8 -*-
import numpy as np
import gym
import gym.spaces

import random
class Key_MDPSettings():
    def __init__(self):        
        self.factor = 3
        self.initial_state = 4
        
        


class Key_Env(MDP):

    def __init__(self):
        
        self.mapping = {0 : (self.apply_up, 'up'),
                        1 : (self.apply_right, 'right'),
                        2 : (self.apply_down, 'down'),
                        3 : (self.apply_left, 'left')}
        self.big_reward = 1
        self.small_reward = .01
        
    def reset(self):
        self.current_state = self.build_state(self.initial_pos)
        self.has_key = False
        
    @property
    def state_size(self): return self.state_space.n 
    @property
    def action_size(self): return self.action_space.n 
    
    def configure(self, cnf):
        self.factor = cnf.factor
        self.shape = (self.factor, self.factor)
        self.initial_pos = cnf.initial_state

        self.action_space = gym.spaces.Discrete(4)
        self.state_space = gym.spaces.Discrete(cnf.factor ** 2)
        self.reset()   
    def is_key_here(self, state):
        i, j = self.get_coords(state)
        return i == 0 and j == self.factor - 1
    def has_ended(self):
        i, j = self.get_coords(self.current_state)
        return i == 0 and j == 0
    def get_template(self, n):
        template = np.zeros(self.shape)
        return template
    
    def build_state(self, n):
        template = self.get_template(n)
        state = template.flatten()
        state[n] = 1
        state = state.reshape(self.shape)
        return state
    def seed(self):
        pass
    def get_coords(self, state):
        [i, j] = np.where(state)
        i, j = i[0], j[0]
        return i, j
    def set_value(self, state, value, i, j):
        state[i][j] = value
        return state
    
    def apply_up(self, old_i, old_j): return max(old_i - 1, 0), old_j
    def apply_down(self, old_i, old_j): return min(old_i + 1, self.factor - 1), old_j        
    def apply_right(self, old_i, old_j): return old_i, min(old_j + 1, self.factor - 1)
    def apply_left(self, old_i, old_j): return old_i, max(old_j - 1, 0)
        
    def move(self, action):
        old_i, old_j = self.get_coords(self.current_state)
        new_i, new_j = self.mapping[action][0](old_i, old_j)
        state = self.set_value(self.current_state, 0, old_i, old_j)
        state = self.set_value(state, 1, new_i, new_j)
        self.current_state = state
        if not self.has_key:
            self.has_key = self.is_key_here(self.current_state)
    
        
    def one_hot_inverse(self, state):
        return np.where(state)[0][0]
        
    def step(self, action):
   
        assert(self.action_space.contains(action))
        self.move(action)
        info = {}
        
        if self.has_ended() and self.has_key:
            reward, terminal = self.big_reward, True
        elif self.has_ended() and not self.has_key: #Explicit is better than implicit
            reward, terminal = self.small_reward, True
        else:
            reward, terminal = 0, False
        observation = self.current_state.flatten()
        return observation, reward, terminal, info
        

# cnf = Key_MDPSettings()
# mdp = Key_Env()
# mdp.configure(cnf)
# mdp.reset()
# s = mdp.current_state
# reward = 0
# t = 0
# while True:
    # a = mdp.action_space.sample()
    # print(reward)
    # print(mdp.current_state)
    # print(mdp.mapping[a][1])
    # _, reward, terminal, info = mdp.step(a)
    # if mdp.has_key and t == 0:
        # t = 1
        # print("********************")
    # if terminal:
        # print(reward)
        # reward = 0
        # print(mdp.current_state)
        # mdp.reset()
        # print("______________")
        # assert t is not 1
    
