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
                        'mode'         : 'train'}
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
                        'env_name'     : 'key_mdp-v0',
                        'scale'        : 5,
                        'factor'       : 3,
                        'log_level'    : 'DEBUG',
                        'display_prob' : 0.05,
                        'use_gpu'      : True,
                        'mode'         : 'train',
                        'architecture' : '100-100-100',
                        'double_q'     : False,
                        'random_seed'  : seed}
                args_list.append(args)
        elif name == 'exp4':
            #HYPERPARAMETER SEARCH
            architectures = [
                    [10],
                    [25, 25],
                    [100],
                    [100, 100]                    
                    ]
            
            hyperparameter_space = {
                    'learning_rate_minimums' : [0.00025, 0.0001],
                    'learning_rates'         : [0.001, 0.1, 0.0005],
                    'architectures'          : architectures
                    }
            base_args = {'agent_type'            : 'dqn',
                    'env_name'              : 'key_mdp-v0',
                    'scale'                 : 25,
                    'factor'                : 3,
                    'log_level'             : 'DEBUG',
                    'display_prob'          : 0.05,
                    'use_gpu'               : True,
                    'mode'                  : 'train',
                    'double_q'              : False,
                    'random_seed'           : 1}
            
            for args in self.get_hyperparameters_iterator(hyperparameter_space,
                                                          base_args):
                print(args)
                if 'architecture' in args:
                    args['architecture'] = '-'.join([str(l) for l in args['architecture']])
                args_list.append(args)
                
        else:
            raise ValueError("Wrong experiment name %s" % name)
            
        self._args_list = args_list
        
    def get_hyperparameters_iterator(self, hyperparameters_space, base_dict):
        lists = []
        for k, hyperparameters in hyperparameters_space.items():
            assert k[-1] == 's'
            param_name = k[:-1]
            list_ = [(param_name, v) for v in hyperparameters]
            
            lists.append(list_)
        for configuration in itertools.product(*lists):
            yield {**base_dict, **dict(configuration)}
    
    def get_args_list(self):
        return self._args_list
