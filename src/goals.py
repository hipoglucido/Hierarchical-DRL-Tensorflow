# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from base import Epsilon
import numpy as np
import math
from configuration import Constants as CT
class Goal(metaclass = ABCMeta):
    def __init__(self, n, name, config = None):
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
            
    def is_achieved(self, screen, action):
        #action not used
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
    def __init__(self, environment, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.environment = environment.gym
        

        
    def get_prep_features(self, observation):
        result = {}
        for feature_name in self.environment.feature_names:
            result[feature_name] = \
                   self.environment.get_prep_feature(observation, feature_name)
        return result
    def is_in_region(self, A_i, A_j, region, n):
        factor = int(math.sqrt(n))
        l = 1 / factor
        matrix = np.zeros(n, dtype = int)
        matrix[region] = 1
        
        matrix = matrix.reshape((factor, factor))
        R_i, R_j = np.where(matrix == 1)
        R_i, R_j = R_i[0], R_j[0]
        R_i_min, R_j_min = R_i * l, R_j * l
        R_i_max, R_j_max = (R_i + 1) * l, (R_j + 1) * l
        i_condition = R_i_min < A_i < R_i_max
        j_condition = R_j_min < A_j < R_j_max
        return i_condition and j_condition
        
    def is_aiming_at(self, A_i, A_j, A_sin, A_cos, B_i, B_j, epsilon = .15):
        
        result = False
        if A_sin > 0:
            A_pointer = math.acos(A_cos)
        else:
            A_pointer = CT.c - math.acos(A_cos)
        
        A_min = (A_pointer - epsilon - CT.c14) % CT.c
        A_max = (A_pointer + epsilon - CT.c14) % CT.c

        i_dist = B_i - A_i
        j_dist = B_j - A_j
#        def r(x): return 360 * x / CT.c

        rel_rad = math.atan2(i_dist, j_dist) + CT.c12

#        print("Ship\t[%.2f, %.2f]" % (r(A_min), r(A_max) ))
#        print("Rela radian", r(rel_rad))
        
        if A_i <= B_i and A_j >= B_j:
#            print(1)
            diff = (CT.c - rel_rad)
            A_target = CT.c12 - diff
        elif A_i >= B_i and A_j >= B_j:
#            print(2)
            A_target = CT.c12 + rel_rad
        elif A_i >= B_i and A_j <= B_j:
#            print(3)            
            A_target = CT.c12 + rel_rad
        elif A_i <= B_i and A_j <= B_j:
#            print(4)
            A_target =  - CT.c12 + rel_rad
        else:
            assert 0
#        print(r(A_target))
        if A_min < A_target < A_max:
            result = True
        elif abs(A_min - A_max) > 2.1 * epsilon:
#            print("wieeerd")
            if A_target > A_min or A_target < A_max:
                result = True
 
        return result
    def is_achieved(self, screen, action):
        pfs = self.get_prep_features(screen)
        
        achieved = False
        if 'aim_at' in self.name:
            if 'square' in self.name:
                # aim_at_aquare
                
                
                if not self.environment.is_no_direction:
                    achieved = self.is_aiming_at(
                                      A_i     = pfs['ship_pos_i'],
                                      A_j     = pfs['ship_pos_j'],
                                      A_sin   = pfs['ship_headings_sin'],
                                      A_cos   = pfs['ship_headings_cos'],
                                      B_i     = pfs['square_pos_i'],
                                      B_j     = pfs['square_pos_j'])
                    


                else:
                    assert 0
            else:
                # aim_at_fortress
                pass
        elif 'region' in self.name:
            _, region_id, total_regions = self.name.split("_")
             
            achieved = self.is_in_region(
                                    A_i      = pfs['ship_pos_i'],
                                    A_j      = pfs['ship_pos_j'],
                                    region = int(region_id),
                                    n      = int(total_regions))
        elif action == CT.SF_action_spaces[self.environment.env_name].index(self.name):
            achieved = True
        return achieved
            
#def generate_area_goals()
 
def generate_SF_goals(environment, goal_names):
    
    goals = {}
    goal_names = [gn for gn in goal_names if gn != 'wait']
    goal_size = len(goal_names) #onehot
    for i, goal_name in enumerate(goal_names):
        print(goal_name)
        goals[i] = SFGoal(n = i,
                          name = goal_name,
                          environment = environment)
        goals[i].setup_one_hot(goal_size)
    #    i = 0
#        for i, action_name in enumerate(CT.SF_action_spaces[environment.env_name]):
#            goals[i] =  SFGoal(
#                            n = i,
#                            name = action_name,
#                            environment = environment)
#        goals[i + 1] = SFGoal(
#                            n = i + 1,
#                            name = 'aim_at_square',
#                            environment = environment)
#    goals = generate_area_goals()
    
    return goals
    
    
    
    
    
    
           
        