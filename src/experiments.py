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
    def __init__(self, name, paralel):
        args_list = []
        if name == 'exp5':
            # pmemory effect on hdqn
            hyperparameter_space = {
                    'pmemorys'               : [0, 1],
                    'random_seeds'           : list(range(20))
                    }
            base_args = {
                    'agent_type'            : 'dqn',
                    'env_name'              : 'SF-v0',
                    'scale'                 : 1000,                   
                    'display_prob'          : 0.05,
                    'use_gpu'               : 0,
                    'mode'                  : 'train',
                    'dueling'               : 0,
                    'double_q'              : 0,
                    'action_repeat'         : 4
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
                    'scale'                 : 1500,
                    'agent_type'            : 'dqn',
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
        elif name == 'exp8':
            #ACTION REPLAY
            hyperparameter_space = {
                    'action_repeats'          : list(range(4,10)),
                    'random_seeds'            : list(range(5))
                    }
            base_args = {
                    'env_name'              : 'SF-v0',
                    'scale'                 :  5000,
                    'agent_type'            : 'hdqn',
                    'log_level'             : 'DEBUG',
                    'display_prob'          :  0.001,
                    'use_gpu'               :  0,
                    'mode'                  :  'train',
                    'architecture'          : [64, 64],
                    'goal_group'            :  3,
                    'dueling'               :  1,
                    'double_q'              :  1 
            }  
        elif name == 'exp9':
            #ACTION REPLAY
            hyperparameter_space = {
                    'agent_types'             : ['dqn', 'hdqn'],
                    'random_seeds'            : list(range(3))
                    }
            base_args = {
                    'env_name'              : 'SF-v0',
                    'scale'                 :  500,
                    'display_prob'          :  0.05,
                    'use_gpu'               :  0,
                    'mode'                  :  'train',
                    'goal_group'            :  3,
                    'dueling'               :  1,
                    'double_q'              :  1,
                    'pmemory'               :  0  
            }
        base_args['paralel'] = paralel
        for args in self.get_hyperparameters_iterator(hyperparameter_space,
                                                      base_args):
            
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
