# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from base import Epsilon
import numpy as np

class Goal(metaclass = ABCMeta):
    def __init__(self, n, name, config):
        self.n = n
        self.name = str(name)
        self.steps_counter = 0.
        self.set_counter = 0.
        self.achieved_counter = 0.        
        
    def setup_epsilon(self, config, start_step):
        
        self._epsilon = Epsilon(config, start_step)
        
    @property
    def epsilon(self):
#        return self._epsilon.mixed_value(self.steps_counter,
#                                    self.set_counter,
#                                    self.achieved_counter)
        
        result = self._epsilon.successes_value(attempts = self.set_counter,
                                            successes = self.achieved_counter)
#        if self.n == 11:
#            print("_________")
#            print(self.set_counter)
#            print(self.achieved_counter)
#            print(result)
#            print("_________")
#            assert result == 1
        return result
    
    def setup_one_hot(self, length):
        one_hot = np.zeros(length)
        one_hot[self.n] = 1.
        self.one_hot = one_hot
    
    @abstractmethod
    def is_achieved(self):
        pass

    
    def finished(self, metrics, is_achieved):
        
        self.achieved_counter += int(is_achieved)
        metrics.store_goal_result(self, is_achieved)
    
class MDPGoal(Goal):
    """
    Goals for the MDP toy problems. In this case the goals are defined as
    reaching a particular state and there are as many goals as possible
    states in the environment
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
            
    def is_achieved(self, screen):
        return np.array_equal(screen, self.one_hot)

class SFGoal(Goal):
    """
    Goals for the Space Fortress games. Ideas:
    
    Goals shared across minigames:
        - *Actions
        - Divide screen in R regions and define R goals as "go to region _"
        
    Goals for the control task (SFC):
        - Aim at the square
        - 
        
    Goals for the shooting task (SFS):    
        - 
    Goals for the "complete" game (SF):
        - Aim at the fortress
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        