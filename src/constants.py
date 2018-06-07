# -*- coding: utf-8 -*-

import math

class Constants:
    

    key_to_sf = {
        'Key.up'     : 65362,
        'Key.right'  : 65363,
        'Key.down'   : 65364,
        'Key.left'   : 65361,
        'Key.space'  : 32,
        'Key.esc'    : -1,
        'wait'       : 0
    }
    
    SF_action_spaces = {
        'SFC-v0'   : ['Key.up', 'Key.right', 'Key.left', 'wait'],
        'SF-v0'    : ['Key.up', 'Key.right', 'Key.left', 'wait', 'Key.space'],
        'SFS-v0'   : [],
        'AIM-v0'   : ['Key.right', 'Key.left', 'wait', 'Key.space']
            }
    SF_envs = list(SF_action_spaces.keys())
    key_to_action = {}
    action_to_sf = {}
    
    for game in SF_envs:
        key_to_action[game] = {str(k) : i for i, k in enumerate(SF_action_spaces[game])}
        action_to_sf[game] = {}
        for i, v in enumerate(SF_action_spaces[game]):
            action_to_sf[game][i] = key_to_sf[str(v)]
    
    SF_observation_space_sizes = {
        'SFC-v0'   : 8,
        'SF-v0'    : 11,
        'SFS-v0'   : 0,
        'AIM-v0'   : 3
            }
#    print(key_to_action)
#    print(action_to_sf)
    
    MDP_envs = ['stochastic_mdp-v0', 'ez_mdp-v0', 'trap_mdp-v0', 'key_mdp-v0']
    GYM_envs = ['CartPole-v0']
    env_names = SF_envs + MDP_envs + GYM_envs
    
    ### GOALS
    def get_region_names(factor):
        total_regions = factor ** 2
        names = ['region_%d_%d' % (i, total_regions) for i in range(total_regions)]
        return names
    
    goal_groups = {
        'SFC-v0' : {
            0 : [],
            1 : get_region_names(4),
            2 : ['aim_at_square'] + get_region_names(4),
            3 : ['aim_at_square'] + SF_action_spaces['SFC-v0'] + get_region_names(4),
            4 : SF_action_spaces['SFC-v0'],
            5 : ['aim_at_square'] + SF_action_spaces['SFC-v0']
            },
        'SF-v0'  : {
            0 : [],
            1 : ['aim_at_fortress']  + SF_action_spaces['SF-v0'] + get_region_names(4),
            2 : ['aim_at_fortress']  + ['Key.space'] + get_region_names(4),
            3 : ['aim_at_fortress', 'aim_at_mine']  + SF_action_spaces['SF-v0'],
            4 : ['aim_at_fortress']  + SF_action_spaces['SF-v0']
                },
        'AIM-v0' : {
            0 : ['aim_at_mine'] + SF_action_spaces['AIM-v0'],
            1 : []
                },
        }  
    
    c = 2 * math.pi
    c34 = 3 / 4 * c
    c12 = 1 / 2 * c
    c14 = 1 / 4 * c
    
    #oneqpi = math.pi * 1 / 4