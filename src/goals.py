# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from epsilon import Epsilon
import collections
import numpy as np
import math
from constants import Constants as CT
import utils


class Goal(metaclass = ABCMeta):
    """
    Abstract class of a goal
    """
    def __init__(self, n, name, config = None):
        self.n = n
        self.name = str(name)
        self.steps_counter = 0.
        self.set_counter = 0.
        self.achieved_counter = 0.
        self.success_rate = 0
        self.steps_for_achievement = 0
        
        if config is not None:
            # Take only the last attempts to compute epsilon
            self.last_attempts = collections.deque(maxlen = \
                                            config.goal_attempts_list_len)
        else:
            # Take all the attempts to compute epsilon
            self.last_attempts = []
        
        self._epsilon = Epsilon()
        
   
    @property
    def epsilon(self):
        """
        The epsilon of a goal is related with its rate of achievement
        """

#        result = self._epsilon.successes_value(
#                            attempts = self.set_counter,
#                            successes = self.achieved_counter)
        try:
            self.success_rate = sum(self.last_attempts) / len(self.last_attempts)
        except ZeroDivisionError:
            self.success_rate = 0
        
        return 1 - min(self.success_rate, .99)

    
    def setup_one_hot(self, length):
        """
        Build the one-hot-encoding representation of the goal, which depends
        on the amount of goals that the hDQN is using
        
        params:
            length: int, total amount of goals that the hDQN is using
        """
        one_hot = np.zeros(length)
        one_hot[self.n] = 1.
        self.one_hot = one_hot
    
    @abstractmethod
    def is_achieved(self):
        pass

    
    def finished(self, metrics, is_achieved):
        """
        Function to indicate that the goal has finished.
        
        params:
            metrics: Metrics, object with metrics to be updated
            is_achieved: Boolean, whether the goal has been achieved or not
        """
        
        self.last_attempts.append(int(is_achieved))
        self.achieved_counter += int(is_achieved)
        metrics.store_goal_result(self, is_achieved)
        self.steps_counter = 0.
    
class MDPGoal(Goal):
    """
    Goals for the MDP toy problems. In this case the goals are defined as
    reaching a particular state and there are as many goals as possible
    states in the environment
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
            
    def is_achieved(self, screen, *args, **kwargs):
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
        self.achieved_inside_frameskip = False
        

        
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
    
    def is_achieved(self, screen, action, info):
        """
        Check if the goal has been achieved taking as information the state
        space, the action and other variables (included in info). The check is
        different for each goal.
        
        return achieved: Boolean, whether the goal has been achieved or not
        """
        pfs = self.get_prep_features(screen)
        
        achieved = False
        if self.achieved_inside_frameskip:
            """
            If the goal has been achieved during frameskip then we take it as
            accomplished without further checking. This only applies to certain
            goals
            """
            achieved = True
        elif self.name == 'G_hit_fortress_twice':
            hit = info['steps_since_last_fortress_hit'] == 0
#            print("detected %d\naux %d\nnormal %d\nmin %d" % (int(hit),
#                                               info['steps_since_last_fortress_hit_aux'],
#                                               info['steps_since_last_fortress_hit'],
#                                               info['min_steps_between_shots']))
            if hit and \
                    info['steps_since_last_fortress_hit_aux'] <= \
                                            info['min_steps_between_shots']:
                achieved = True
        if self.name == 'G_hit_fortress_once':
            if info['min_steps_between_shots'] == info['steps_since_last_fortress_hit']:
                achieved = True
        if self.name == 'G_single_shoot':
            if info['min_steps_between_shots'] <= info['steps_since_last_shot'] \
                <= max(info['action_repeat'], info['min_steps_between_shots']):
                achieved = True                
        elif self.name == 'G_double_shoot':
            its_a_shot = action == \
                CT.SF_action_spaces[self.environment.env_name].index('Key.space')            
            if its_a_shot and \
               0 <= info['steps_since_last_shot'] < info['min_steps_between_shots']:
                achieved = True
        elif self.name == 'G_hit_mine':
            if info['mine_hit']:
                achieved = True
            elif pfs['mine_pos_i'] == 0 and pfs['mine_pos_j'] == 0:
                achieved = True
        elif self.name == 'G_shoot_at_fortress':
            if info['fortress_hit']:
                achieved = True
        elif 'aim_at' in self.name:
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
                epsilon = .2        
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
                    if pfs['mine_pos_i'] == 0 and pfs['mine_pos_j'] == 0:
                        """
                        If there is no mine in the screen we set the goal to
                        True so that the agent doesn't get stuck trying to
                        achieve something that is impossible (at least until
                        one mine appears)
                        """
                        achieved = True
                    else:
                        achieved = self.is_aiming_at(
                                          A_i     = new_pfs['ship_pos_i'],
                                          A_j     = new_pfs['ship_pos_j'],
                                          A_sin   = pfs['ship_headings_sin'],
                                          A_cos   = pfs['ship_headings_cos'],
                                          B_i     = pfs['mine_pos_i'],
                                          B_j     = pfs['mine_pos_j'],
                                          epsilon = epsilon)
            elif 'fortress' in self.name:
                epsilon = .15
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
            _, _, region_id, total_regions = self.name.split("_")
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
        elif self.name[2:] in CT.SF_action_spaces[self.environment.env_name]:
            # Low level goals (aka actions)
            goal_action = CT.SF_action_spaces[self.environment.env_name].index(self.name[2:])
            achieved = action == goal_action
        
        return achieved
            

 
def generate_SF_goals(environment, goal_names, config):
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
                          name        = goal_name,
                          environment = environment,
                          config      = config)
        goals[i].setup_one_hot(goal_size)

    return goals
    
    
    
    
    
    
           
        
