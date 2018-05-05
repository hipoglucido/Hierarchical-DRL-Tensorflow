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
    def __init__(self, environment, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.environment = environment.gym
        

        
    def get_prep_features(self, observation):
        result = {}
        for feature_name in self.environment.feature_names:
            result[feature_name] = \
                   self.environment.get_prep_feature(observation, feature_name)
        return result
    def is_aiming_at(self, A_i, A_j, A_sin, A_cos, B_i, B_j, epsilon = .1):
        
        result = False
#        ship_headings_sin =pfs['ship_headings_sin']
#        ship_headings_cos =pfs['ship_headings_cos']
        if A_sin > 0:
            A_pointer = math.acos(A_cos)# / (2*math.pi)
        else:
            A_pointer = CT.c - math.acos(A_cos)# / (2*math.pi)
        #rad /= 2 * math.pi
        A_min = (A_pointer - epsilon - CT.c14) % CT.c
        A_max = (A_pointer + epsilon - CT.c14) % CT.c
#                    g = 360 * rad / (2 * math.pi)
        i_dist = B_i - A_i
        j_dist = B_j - A_j
        def r(x): return 360 * x / CT.c
#                    abs_dist = math.sqrt( abs(i_dist) ** 2 + abs(j_dist) ** 2)
#                    tan_alpha = i_dist / j_dist
        rel_rad = math.atan2(i_dist, j_dist) + CT.c12
#                    ship_rad = (ship_rad + math.pi) % (2 * math.pi)
#                    target_rad = target_rad + math.pi/2 % 2 * math.pi
#                    alpha = math.atan(tan_alpha)
#                    dist_angle = alpha - g
        
#        print("frel", r(frel))
#                    error = target_rad - ship_rad
#        diff = abs(A_min - A_max)
        print("Ship\t[%.2f, %.2f]" % (r(A_min), r(A_max) ))
        print("Rela radian", r(rel_rad))
        
        if A_i <= B_i and A_j >= B_j:
            print(1)
            diff = (CT.c - rel_rad)
            A_target = CT.c12 - diff
        elif A_i >= B_i and A_j >= B_j:
            print(2)
            A_target = CT.c12 + rel_rad
        elif A_i >= B_i and A_j <= B_j:
            print(3)
            
            A_target = CT.c12 + rel_rad
        elif A_i <= B_i and A_j <= B_j:
            print(4)
            A_target =  - CT.c12 + rel_rad
        else:
            print("MAAAAAAAAL")
        print(r(A_target))
        
      
        if A_min < A_target < A_max:
            result = True
        elif abs(A_min - A_max) > 2.1 * epsilon:
            print("wieeerd")
            if A_target > A_min or A_target < A_max:
                result = True
            
#        error = -1
#        is_weird = diff > beam_size*1.1
#        if is_weird:
#            print("WEIRD")
#        if CT.c34 <= rel_rad <= CT.c:                        
#            print(1)
#                
#            if not any([CT.c14 < A_min < CT.c12,
#                       CT.c14 < A_max < CT.c12]):
#                result = False
#            else:
#                print("Das",r((rel_rad)))
#                result = A_min < (rel_rad + CT.c12) < A_max
#        elif 0 <= rel_rad <= CT.c14:
#            print(2)   
#            if is_weird:
#                result = False
#            else:
#                t = (rel_rad + CT.c12) % CT.c
#                result = A_min < t < A_max                 
#        elif CT.c14 <= rel_rad <= CT.c12:
#            print(3)
#            if is_weird:
#                t = (rel_rad + CT.c12) % CT.c
#            else:
#                t = (rel_rad + CT.c12) % CT.c
#                result = A_min < t < A_max
#            print(99,t)
#        elif CT.c12 <= rel_rad <= CT.c34:
#            print(4)
#        else:
#            print("MAL")
        
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
        elif action == CT.SF_action_spaces[self.environment.env_name].index(self.name):
            achieved = True
        return achieved
            

    
def generate_SF_goals(environment):
    goals = []
    i = 0
#    for i, action_name in enumerate(CT.SF_action_spaces[environment.env_name]):
#        goals.append(
#                SFGoal(
#                    n = i,
#                    name = action_name,
#                    environment = environment))
    goals.append(
            SFGoal(
                n = i + 1,
                name = 'aim_at_square',
                environment = environment))
    
    return goals
    
    
    
    
    
    
           
        