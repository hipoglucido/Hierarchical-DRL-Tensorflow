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
        'SF-v0'    : 12,
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
    move_actions = ['Key.up', 'Key.right', 'Key.left', 'wait']
    """
    Only low level resembles DQN?
    Can it pick the best goals if goal set is very big?
    Is aiming important?
    Will he use Key.space as goal if it has shoot goals?
    """
    goal_groups = {
        # Control task
        'SFC-v0' : {
            0 : [],
            1 : get_region_names(4),
            2 : ['aim_at_square'] + get_region_names(4),
            3 : ['aim_at_square'] + SF_action_spaces['SFC-v0'] + get_region_names(4),
            4 : SF_action_spaces['SFC-v0'],
            5 : ['aim_at_square'] + SF_action_spaces['SFC-v0']
            },
    
        # SF task
        #get_region_names(5)
        # 'Key.up', 'Key.right', 'Key.left', 'wait', 'Key.space'
        'SF-v0'  : {
            # Empty
            0 : [],
            # Only low level actions
            1 : SF_action_spaces['SF-v0'],
            # Only high level
            2 : ['aim_at_fortress', 'aim_at_mine', 'shoot_at_mine',
                 'shoot_at_fortress'] + get_region_names(5),
            # Everything but no Key.space
            3 : ['aim_at_fortress', 'aim_at_mine', 'shoot_at_mine',
                 'shoot_at_fortress'] + move_actions,
            # Without aiming    
            4 : ['shoot_at_mine', 'shoot_at_fortress'] + move_actions,
            # Is Key.space used?
            5 : ['shoot_at_mine', 'shoot_at_fortress'] + SF_action_spaces['SF-v0'],
            
            6 : ['aim_at_fortress', 'aim_at_mine'] + SF_action_spaces['SF-v0'],
    
                },
        # Aim task
        'AIM-v0' : {
            0 : ['aim_at_mine'] + SF_action_spaces['AIM-v0'],
            1 : []
                },
        }
    
    temp = {}
    for env_name, goals_dict in goal_groups.items():
        temp[env_name] = {}
        for key, goals in goals_dict.items():
            temp[env_name][key] = ['G_' + g for g in goals]
    goal_groups = temp
    
    c = 2 * math.pi
    c34 = 3 / 4 * c
    c12 = 1 / 2 * c
    c14 = 1 / 4 * c
    
