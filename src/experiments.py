# -*- coding: utf-8 -*-
import subprocess
import itertools
import os
import utils
import pprint
def run_cmd(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    output, error = process.communicate()
    return output
class Experiment():
    def __init__(self, name, paralel):
        args_list = []
        if name == 'exp1':
            # toy_problem
            hyperparameter_space = {
                    'agent_types'             : ['dqn', 'hdqn'],
                    'random_seeds'           : list(range(10))
                    }
            base_args = {
                    'agent_type'            : 'dqn',
                    'env_name'              : 'key_mdp-v0',
                    'scale'                 : 100,    
                    'mode'                  : 'train',
                    'display_prob'          : 0.001,
                    'use_gpu'               : 0,
                    'dueling'               : 0,
                    'double_q'              : 0
            }  
        elif name == 'exp2':
            # DOUBLE_Q
            hyperparameter_space = {
                    'double_qs'               : [0, 1],
                    'random_seeds'           : list(range(5))
                    }
            base_args = {
                    'agent_type'            : 'dqn',
                    'scale'                 : 5000,  
                    'env_name'              : 'SF-v0', 
                    'ez'                    : 1,
                    'use_gpu'               : 0,
                    'pmemory'               : 0,
                    'mines_activated'       : 1,
                    'dueling'               : 0,
                    'action_repeat'         : 4
            }  
        elif name == 'exp3':
            # DUELING
            hyperparameter_space = {
                    'duelings'               : [0, 1],
                    'random_seeds'           : list(range(5))
                    }
            base_args = {
                    'agent_type'            : 'dqn',
                    'scale'                 : 5000,  
                    'env_name'              : 'SF-v0',
                    'ez'                    : 1,  
                    'use_gpu'               : 0,
                    'pmemory'               : 0,
                    'mines_activated'       : 1,
                    'double_q'              : 0,
                    'action_repeat'         : 4
            }  
        elif name == 'exp4':
            # PRIORITIZED EXPERIENCE REPLAY
            hyperparameter_space = {
                    'pmemorys'               : [0, 1],
                    'random_seeds'           : list(range(5))
                    }
            base_args = {
                    'agent_type'            : 'dqn',
                    'scale'                 : 5000,  
                    'env_name'              : 'SF-v0',
                    'ez'                    : 1,  
                    'use_gpu'               : 0,
                    'dueling'               : 0,
                    'mines_activated'       : 1,
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
                    'agent_types'             : ['dqn', 'hdqn']
                    }
            base_args = {
                    'env_name'              : 'SF-v0',
                    'scale'                 :  500,
                    'display_prob'          :  0.03,
                    'use_gpu'               :  1,
                    'mode'                  :  'train',
                    'goal_group'            :  3,
                    'dueling'               :  1,
                    'double_q'              :  1,
                    'pmemory'               :  0  
            }  
        elif name == 'exp10':
            architectures = [[100],
                             [25, 25],
                             [256, 256]]
            
            
            #misc
            hyperparameter_space = {
                    'architectures'         : architectures,
                    'duelings'              : [0, 1],
                    'double_qs'             : [0, 1],
                    'pmemorys'              : [0, 1]
                    
                    }
            base_args = {
                    'env_name'              : 'SF-v0',
                    'agent_type'            : 'dqn',
                    'scale'                 :  10000,
                    'display_prob'          :  0,
                    'use_gpu'               :  0,
                    'mode'                  :  'train',
                    'mines_activated'       :  0,
                    'goal_group'            :  3
            }
         
        print("Experiment %s:\n%s" % (name, pprint.pformat(hyperparameter_space)))
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
            assert k[-1] == 's', '"%s" must end with "s"' % k
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
