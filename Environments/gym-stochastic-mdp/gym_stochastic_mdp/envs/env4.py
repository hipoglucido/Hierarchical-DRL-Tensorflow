# -*- coding: utf-8 -*-
import numpy as np
import gym
import gym.spaces
from mdp import MDP
import random

        


class Key_MDPEnv(MDP):

    def __init__(self):
        
        self.mapping = {0 : (self.apply_up, 'up'),
                        1 : (self.apply_right, 'right'),
                        2 : (self.apply_down, 'down'),
                        3 : (self.apply_left, 'left')}
        self.big_reward = 1
        self.small_reward = 0
        self.negative_reward = -1
        
    def reset(self):
        if not self.random_reset:
            self.current_state = self.build_state(self.initial_pos)
        else:
            valid = False
            while not valid:
                n_state = random.randint(0, self.state_size - 1)
                state = self.build_state(n_state)
                i, j = self.get_coords(state)
                invalid = [self.is_upper_left, self.is_upper_right,
                           self.is_bottom_left, self.is_bottom_right]
                if not any([pos(i, j) for pos in invalid]):
                    valid = True
                    self.current_state = state
                
                
        self.has_key = False
        observation = self.current_state.flatten()
        return observation
                
            
        
    @property
    def state_size(self): return self.state_space.n 
    @property
    def action_size(self): return self.action_space.n 
    
    def configure(self, cnf):
        self.factor = cnf.env.factor
        self.shape = (self.factor, self.factor)
        self.initial_pos = cnf.env.initial_state

        self.action_space = gym.spaces.Discrete(4)
        self.state_space = gym.spaces.Discrete(self.factor ** 2)
        self.random_reset = cnf.env.random_reset
        self.reset()   
    def is_key_here(self, state):
        i, j = self.get_coords(state)
        #Bottom right corner
        return self.is_bottom_right(i, j)
    
    def is_upper_left(self, i, j): 
        return i == 0 and j == 0
    def is_bottom_right(self, i, j):
        return i == self.factor - 1 and j == self.factor - 1
    def is_upper_right(self, i, j):
        return i == 0 and j == self.factor - 1
    def is_bottom_left(self, i, j):
        return i == self.factor - 1 and j == 0
    
    def has_ended(self):
        i, j = self.get_coords(self.current_state)
        return any([self.is_upper_left(i, j), self.is_upper_right(i, j),
                    self.is_bottom_left(i, j), self.is_bottom_right(i, j)])
        
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
    
    #TODO reorganize so that it doesn't take parameters but also don't calculate
    # coordinates again and again
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
            if self.has_key:
                pass#print("KEY FOUND!")
        return new_i, new_j
        
    def one_hot_inverse(self, state):
        return np.where(state)[0][0]
        #return state.reshape(self.shape)
        
    def step(self, action):
        assert(self.action_space.contains(action))
        new_i, new_j = self.move(action)
        info = {}
        if self.is_upper_right(new_i, new_j) or self.is_bottom_left(new_i, new_j):
            reward, terminal = self.negative_reward, True
        elif self.is_upper_left(new_i, new_j) and self.has_key:
            reward, terminal = self.big_reward, True
        elif self.is_upper_left(new_i, new_j) and not self.has_key:
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
    
