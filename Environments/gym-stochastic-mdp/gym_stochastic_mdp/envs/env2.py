import random
import numpy as np
from mdp import MDP

#
#TOTAL_STATES = 20
#INITIAL_STATE = 1
#TERMINAL_STATES = [0, TOTAL_STATES - 1]
#TOTAL_ACTIONS = 2
#RIGHT_FAILURE_PROB = 0.1
BIG_REWARD = 1.
SMALL_REWARD = 1./ 100.
class EZ_MDPEnv(MDP):
    def __init__(self):
        pass
    def configure(self, cnf):
        self.initial_state = cnf.initial_state
        super().configure(total_states = cnf.total_states,
                         total_actions = cnf.total_actions,
                         terminal_states = cnf.terminal_states)
        self.right_failure_prob = cnf.right_failure_prob
    
        
    def step(self, action):
        if self.has_ended():
            raise RuntimeError("Environment already ended")
        
        assert(self.action_space.contains(action))
        
        info = dict()
        if action == 1 and random.random() > self.right_failure_prob:
            s = self.current_state + 1
        else:
            s = self.current_state - 1    

        if self.state_space.contains(s):
            self.current_state = s
        else:
            #Don't move
            # e.g. top right state + right = top right state
            pass
        if not self.has_ended():
            reward, done = 0, False    
        elif self.current_state == self.state_size - 1:
            reward, done = BIG_REWARD, True
            
        else:
            reward, done = SMALL_REWARD, True
        one_hot_state = self.one_hot(self.current_state)
        
        return one_hot_state, reward, done, info
        

