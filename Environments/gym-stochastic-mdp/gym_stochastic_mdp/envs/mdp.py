from abc import ABCMeta, abstractmethod
import numpy as np
import gym
import gym.spaces

class MDP(gym.Env, metaclass=ABCMeta):
    
    def __init__(self):
        pass
    
    def configure(self, total_states, total_actions, terminal_states):
        self.action_space = gym.spaces.Discrete(total_actions)
        self.state_space = gym.spaces.Discrete(total_states)
        self.observation_space = self.state_space
        self.terminal_states = terminal_states
        self.reset()
    
    
    @abstractmethod
    def step(self, action):
        pass

    @property
    def action_size(self):
        return self.action_space.n
    
    @property
    def observation_size(self):
        return self.state_size
    
    @property
    def state_size(self):
        return self.state_space.n
    
    def has_ended(self):
        result = self.current_state in self.terminal_states    
        return result

    def reset(self):
        self.current_state = self.initial_state
        one_hot_state = self.one_hot(self.current_state)
        return one_hot_state
    
    def render(self):
        print("State:",self.one_hot(self.current_state), self.current_state)
        pass
    
    def seed(self):
        return
    
    def one_hot(self, state):
        vector = np.zeros(self.state_size)
        vector[state] = 1.0
        #return np.expand_dims(vector, axis=0)    
        return vector
    
    def one_hot_inverse(self, state):
        return np.where(state)[0][0]
    
    def lives(self): return None
        
    def random_test(self):    
        while 1:
            #print("____________")
            self.render()
            a = self.action_space.sample()
            _, r, t, _ = self.step(a)
            print(a)
            #self.render()
            if t:
                self.render()
                print("***Terminal",r)
                
                self.reset()
                if r == 1:
                    pass
            else:
                assert r==0
    def close(self):
        raise NotImplemented
        















    
    