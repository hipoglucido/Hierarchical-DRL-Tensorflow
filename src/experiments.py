# -*- coding: utf-8 -*-
import subprocess
import os
import utils
def run_cmd(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    output, error = process.communicate()
    return output
class Experiment():
    def __init__(self, name):
        if name == 'exp1':
            args_list = []
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
            
            self._args_list = args_list
        elif name == 'exp2':
            args_list = []
            
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
            
            self._args_list = args_list
            
        else:
            raise ValueError("Wrong experiment name %s" % name)
    def get_args_list(self):
        return self._args_list
