# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from base import Epsilon
import numpy as np
import math
from configuration import Constants as CT
import utils
class Goal(metaclass = ABCMeta):
    def __init__(self, n, name, config = None):
        self.n = n
        self.name = str(name)
        self.steps_counter = 0.
        self.set_counter = 0.
        self.achieved_counter = 0.
        
        self._epsilon = Epsilon()
        
#    def setup_epsilon(self, config, start_step):
#        
#        self._epsilon = Epsilon(start_value = 1.,
#                                  end_value   = .05,
#                                  start_t     = start_step,
#                                  end_t       = config.max_step,
#                                  learn_start = 1000)
#        
    @property
    def epsilon(self):

        result = self._epsilon.successes_value(
                            attempts = self.set_counter,
                            successes = self.achieved_counter)

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

        A_pointer = utils.revert_cyclic_feature(X_sin       = A_sin,
                                              X_cos       = A_cos,
                                              is_scaled   = True,
                                              scale_after = False)

        result = False
     
        A_min = (A_pointer - epsilon - CT.c14) % CT.c
        A_max = (A_pointer + epsilon - CT.c14) % CT.c
        #A_mean = (A_pointer - CT.c14) % CT.c
        i_dist = B_i - A_i
        j_dist = B_j - A_j


        rel_rad = math.atan2(i_dist, j_dist) + CT.c12
        

        if A_i <= B_i and A_j >= B_j:
            # upper right
            diff = (CT.c - rel_rad)
            A_target = CT.c12 - diff
        elif A_i >= B_i and A_j >= B_j:
            # bottom right
            A_target = CT.c12 + rel_rad
        elif A_i >= B_i and A_j <= B_j:
            # bottom left      
            A_target = CT.c12 + rel_rad
        elif A_i <= B_i and A_j <= B_j:
            # upper left
            A_target =  - CT.c12 + rel_rad
        else:
            assert 0

        weird = abs(A_min - A_max) > 2.1 * epsilon

        if A_min < A_target < A_max:
            result = True
        elif weird:
            
            if A_target > A_min or A_target < A_max:
                result = True
 
        return result
    def is_achieved(self, screen, action):
        
        pfs = self.get_prep_features(screen)
        
        achieved = False
        if 'aim_at' in self.name:
            if self.environment.is_no_direction:
                    #Aiming at square doesn't make sense if rotation is deactivated
                    assert 0
            if 'square' in self.name:
                # aim_at_aquare
                
                epsilon = 0.3
                if self.environment.is_wrapper:
                    # Rotation activated and WRAPPING
                    new_pfs = {}
                    coordinate_fns = ['ship_pos_i', 'ship_pos_j', 'square_pos_i',
                                      'square_pos_j']
                    for fn in coordinate_fns:
                        new_pfs[fn] = utils.revert_cyclic_feature(
                                X_sin         = pfs[fn + '_sin'],
                                X_cos         = pfs[fn + '_cos'],
                                is_scaled     = True,
                                scale_after   = True)
                    
                    achieved = self.is_aiming_at(
                                      A_i     = new_pfs['ship_pos_i'],
                                      A_j     = new_pfs['ship_pos_j'],
                                      A_sin   = pfs['ship_headings_sin'],
                                      A_cos   = pfs['ship_headings_cos'],
                                      B_i     = new_pfs['square_pos_i'],
                                      B_j     = new_pfs['square_pos_j'],
                                      epsilon = epsilon)
                    
                else:
                    # Rotation activated and NO WRAPPING
                    achieved = self.is_aiming_at(
                                      A_i     = pfs['ship_pos_i'],
                                      A_j     = pfs['ship_pos_j'],
                                      A_sin   = pfs['ship_headings_sin'],
                                      A_cos   = pfs['ship_headings_cos'],
                                      B_i     = pfs['square_pos_i'],
                                      B_j     = pfs['square_pos_j'],
                                      epsilon = epsilon)

                
            elif 'mine' in self.name:
                # aim_at_mine
                achieved = self.is_aiming_at(
                                      A_i     = .5,
                                      A_j     = .5,
                                      A_sin   = pfs['ship_headings_sin'],
                                      A_cos   = pfs['ship_headings_cos'],
                                      B_i     = pfs['mine_pos_i'],
                                      B_j     = pfs['mine_pos_j'],
                                      epsilon = epsilon)
            elif 'fortress' in self.name:
                epsilon = .1
                if self.environment.is_wrapper:
                    # Rotation activated and WRAPPING
                    new_pfs = {}
                    coordinate_fns = ['ship_pos_i', 'ship_pos_j']
                    for fn in coordinate_fns:
                        new_pfs[fn] = utils.revert_cyclic_feature(
                                X_sin         = pfs[fn + '_sin'],
                                X_cos         = pfs[fn + '_cos'],
                                is_scaled     = True,
                                scale_after   = True)
                    
                    achieved = self.is_aiming_at(
                                      A_i     = new_pfs['ship_pos_i'],
                                      A_j     = new_pfs['ship_pos_j'],
                                      A_sin   = pfs['ship_headings_sin'],
                                      A_cos   = pfs['ship_headings_cos'],
                                      B_i     = .5,
                                      B_j     = .5,
                                      epsilon = epsilon)
                else:
                    # Rotation activated and NO WRAPPING
                    achieved = self.is_aiming_at(
                                      A_i     = pfs['ship_pos_i'],
                                      A_j     = pfs['ship_pos_j'],
                                      A_sin   = pfs['ship_headings_sin'],
                                      A_cos   = pfs['ship_headings_cos'],
                                      B_i     = .5,
                                      B_j     = .5,
                                      epsilon = epsilon)
                
            else:
                assert 0
        elif 'region' in self.name:
            _, region_id, total_regions = self.name.split("_")
            if self.environment.is_wrapper:
                new_pfs = {}
                coordinate_fns = ['ship_pos_i', 'ship_pos_j']
                for fn in coordinate_fns:
                    new_pfs[fn] = utils.revert_cyclic_feature(
                            X_sin         = pfs[fn + '_sin'],
                            X_cos         = pfs[fn + '_cos'],
                            is_scaled     = True,
                            scale_after   = True)
                achieved = self.is_in_region(
                                        A_i      = new_pfs['ship_pos_i'],
                                        A_j      = new_pfs['ship_pos_j'],
                                        region = int(region_id),
                                        n      = int(total_regions))
            else:
                achieved = self.is_in_region(
                                        A_i      = pfs['ship_pos_i'],
                                        A_j      = pfs['ship_pos_j'],
                                        region = int(region_id),
                                        n      = int(total_regions))
        elif self.name in CT.SF_action_spaces[self.environment.env_name]:
            goal_action = CT.SF_action_spaces[self.environment.env_name].index(self.name)
            achieved = action == goal_action
            
        elif ' escape_from_fortress':
            if self.environment.is_wrapper:
                # Rotation activated and WRAPPING
                new_pfs = {}
                coordinate_fns = ['ship_pos_i', 'ship_pos_j']
                for fn in coordinate_fns:
                    new_pfs[fn] = utils.revert_cyclic_feature(
                            X_sin         = pfs[fn + '_sin'],
                            X_cos         = pfs[fn + '_cos'],
                            is_scaled     = True,
                            scale_after   = True)
                
                achieved = self.is_aiming_at(
                                  A_i     = .5,
                                  A_j     = .5,
                                  A_sin   = pfs['ship_headings_sin'],
                                  A_cos   = pfs['ship_headings_cos'],
                                  B_i     = new_pfs['ship_pos_i'],
                                  B_j     = new_pfs['ship_pos_j'],
                                  epsilon = .3)
                
            else:
                # Rotation activated and NO WRAPPING
                achieved = self.is_aiming_at(
                                  A_i     = .5,
                                  A_j     = .5,
                                  A_sin   = pfs['ship_headings_sin'],
                                  A_cos   = pfs['ship_headings_cos'],
                                  B_i     = pfs['ship_pos_i'],
                                  B_j     = pfs['ship_pos_j'],
                                  epsilon = .3)   
        return achieved
            

 
def generate_SF_goals(environment, goal_names):
    """
    Gnerate Goal objects
    
    params:
        environment: gym object
        goal_names: list of strings with the name of the goals
    """
    goals = {}
    goal_names_to_exclude = []#['wait']
    goal_names = [gn for gn in goal_names if gn not in goal_names_to_exclude]
    goal_size = len(goal_names) #onehot
    for i, goal_name in enumerate(goal_names):
        #print(goal_name)
        goals[i] = SFGoal(n = i,
                          name = goal_name,
                          environment = environment)
        goals[i].setup_one_hot(goal_size)

    return goals
    
    
    
    
    
    
           
        
