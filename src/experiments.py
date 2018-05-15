# -*- coding: utf-8 -*-
import subprocess
import itertools
import os
import utils
def run_cmd(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    output, error = process.communicate()
    return output
class Experiment():
    def __init__(self, name):
        args_list = []
        if name == 'exp1':
            #ARCHITECTURES
            archs = [
                    [10, 50, 100],
                    [10],
                    [10, 10, 10],
                    [200],
                    [200, 200, 200] 
                    ]
            for arch in archs:
                args = {'agent_type'   : 'dqn',
                        'env_name'     : 'key_mdp-v0',
                        'scale'        : 4,
                        'factor'       : 3,
                        'log_level'    : 'DEBUG',
                        'display_prob' : 0.05,
                        'use_gpu'      : True,
                        'mode'         : 'train'
                        }
                args['architecture'] = '-'.join([str(l) for l in arch])
                args_list.append(args)
        elif name == 'exp2':
            #DOUBLE Q
            
            for double_q in [False, True]:
                args = {'agent_type'   : 'dqn',
                        'env_name'     : 'key_mdp-v0',
                        'scale'        : 10,
                        'factor'       : 5,
                        'log_level'    : 'DEBUG',
                        'display_prob' : 0.05,
                        'use_gpu'      : True,
                        'mode'         : 'train',
                        'architecture' : '100',
                        'double_q'     : double_q}
                args_list.append(args)
            
        elif name == 'exp3':
            #RANDOM SEEDS
            args_list = []
            seeds = range(1, 10)
            for seed in seeds:
                args = {'agent_type'   : 'dqn',
                        'env_name'     : 'SFC-v0',
                        'scale'        : 10000,
                        'factor'       : 3,
                        'memory_size'  : 500000,
                        'log_level'    : 'DEBUG',
                        'display_prob' : 0.03,
                        'use_gpu'      : 1,
                        'pmemory'      : 1,
                        'mode'         : 'train',
                        'architecture' : '25-25',
                        'double_q'     : 1,
                        'dueling'      : 1,
                        'random_seed'  : seed}
                args_list.append(args)
        elif name == 'exp4':
            #HYPERPARAMETER SEARCH
            architectures = [
#                    [32, 32, 32, 32],
#                    [128],
                    [64, 64],
#                    [4],
#                    [128]
#                    [64, 64, 64]
#                    [10, 10, 10]
#                    [25, 25]                  
                    ]
            duelings = [
                    1
#                    0
                    ]
            pmemorys = [
#                    1,
                    0
                    ]
            double_qs = [
                    1,
#                    0
                    ]
            memory_sizes = [
                    1000000
                    ]
            action_repeats = [
#                    2,
#                    3,
#                    4,
#                    5,
                    6
                    ]
            random_seeds = list(range(2))
            hyperparameter_space = {
#                    'learning_rate_minimums' : [0.00025, 0.0001],
#                    'learning_rates'         : [0.001, 0.1, 0.0005],
                    'architectures'          : architectures,
                    'duelings'               : duelings,
                    'double_qs'              : double_qs,
                    'pmemorys'               : pmemorys,
                    'memory_sizes'           : memory_sizes,
                    'random_seeds'           : random_seeds,
                    'action_repeats'         : action_repeats
                    }
            base_args = {'agent_type'            : 'dqn',
                    'env_name'              : 'SFC-v0',
                    'scale'                 : 250,
                    'factor'                : 3,
                    'log_level'             : 'DEBUG',
                    'display_prob'          : 0.001,
                    'use_gpu'               : 1,
                    'mode'                  : 'train',
#                    'double_q'              : False
            } 
        elif name == 'exp5':
            # pmemory effect on hdqn
            hyperparameter_space = {
                    'pmemorys'               : [0, 1],
                    'random_seeds'           : [0, 1, 2]
                    }
            base_args = {
                    'agent_type'            : 'hdqn',
                    'env_name'              : 'key_mdp-v0',
                    'scale'                 : 200,
                    'factor'                : 3,
                    'log_level'             : 'DEBUG',
                    'display_prob'          : 0.001,
                    'use_gpu'               : 1,
                    'mode'                  : 'train',
                    'architecture'          : [25, 25],
                    'dueling'               : 0,
                    'double_q'              : 0,
                    'action_repeat'         : 1,
                    'memory_size'           : 1000000
#                    'double_q'              : False
            }  
        elif name == 'exp6':
            #hdqn vs dqn in SF
            hyperparameter_space = {
                    'agent_types'             : ['dqn', 'hdqn'],
                    'action_repeats'          : [4]
                    }
            base_args = {
                    'env_name'              : 'SF-v0',
                    'scale'                 : 500,
                    'log_level'             : 'DEBUG',
                    'display_prob'          : 0.001,
                    'use_gpu'               : 1,
                    'mode'                  : 'train',
                    'architecture'          : [64, 64],
                    'dueling'               : 1,
                    'double_q'              : 1,
                    'memory_size'           : 1000000
            }    
        elif name == 'exp7':
            #hdqn vs dqn in SF
            hyperparameter_space = {
                    'goal_groups'             : [1, 2, 3],
                    'random_seeds'            : [7, 14]
                    }
            base_args = {
                    'env_name'              : 'SF-v0',
                    'scale'                 : 500,
                    'agent_type'            : 'hdqn',
                    'log_level'             : 'DEBUG',
                    'display_prob'          : 0.001,
                    'use_gpu'               : 1,
                    'mode'                  : 'train',
                    'architecture'          : [64, 64],
                    'dueling'               : 1,
                    'double_q'              : 1,
                    'memory_size'           : 5000000,
                    'action_repeat'         : 4 
            }           
        for args in self.get_hyperparameters_iterator(hyperparameter_space,
                                                      base_args):
            print(args)
            if 'architecture' in args:
                args['architecture'] = '-'.join([str(l) for l in args['architecture']])
            args_list.append(args)
                
    
            
        self._args_list = args_list
        
    def get_hyperparameters_iterator(self, hyperparameters_space, base_dict):
        lists = []
        for k, hyperparameters in hyperparameters_space.items():
            assert k[-1] == 's'
            param_name = k[:-1]
            list_ = [(param_name, v) for v in hyperparameters]
            
            lists.append(list_)
        import random
        configurations = list(itertools.product(*lists))
        random.shuffle(configurations)
        for configuration in configurations:
            yield {**base_dict, **dict(configuration)}
    
    def get_args_list(self):
        return self._args_list
