from __future__ import print_function
import logging
import random
import tensorflow as tf

from environment import Environment


import configuration
from configuration import Constants as CT

import utils
import sys
import argparse
from pprint import pformat
from hDQN_agent import HDQNAgent
from DQN_agent import DQNAgent

parser = argparse.ArgumentParser()


# GLOBAL PARAMETERS
gl_args = parser.add_argument_group('Global')
gl_args.add_argument("--use_gpu", default = None, type = utils.str2bool)
gl_args.add_argument("--gpu_fraction", default = None)
gl_args.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
gl_args.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Log level.")
gl_args.add_argument("--display_prob", default = None, type = float)

# ENVIRONMENT PARAMETERS
env_args = parser.add_argument_group('Environment')
env_args.add_argument("--mdp_prob", default = None)
env_args.add_argument("--env_name", choices = CT.env_names ,default = "ez_mdp-v0")
env_args.add_argument("--right_failure_prob", default = None)
# AGENT PARAMETERS
ag_args = parser.add_argument_group('Agent')
ag_args.add_argument("--scale", default = 50, type = int)
ag_args.add_argument("--agent_type", default = None, type = str)
ag_args.add_argument("--train", default = None, type = bool)

#
args = vars(parser.parse_args())

# Set random seed
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            level = getattr(logging, args['log_level']))

  
cnf = configuration.Configuration()

#Global settings
gl_st = configuration.GlobalSettings(args)

cnf.set_global_settings(gl_st)

#Agent settings
if args['agent_type'] == 'dqn':
    ag_st = configuration.DQNSettings(args['scale'])
elif args['agent_type'] == 'hdqn':
    ag_st = configuration.hDQNSettings(args['scale'])
else:
    raise ValueError("Wrong agent")
    
ag_st.update(args)
cnf.set_agent_settings(ag_st)

#Environment settings
utils.insert_dirs(cnf.gl.env_dirs)


if args['env_name'] in CT.SF_envs:
    #Space Fortress
    env_st = configuration.SpaceFortressSettings(new_attrs = args)
    
elif args['env_name'] in CT.MDP_envs:
    #MDP
    env_st = configuration.MDPSettings(new_attrs = args)
else:
    raise ValueError("Wrong env_name %s".format(args['env_name']))

cnf.set_environment_settings(env_st)
environment = Environment(cnf)



tf.set_random_seed(gl_st.random_seed)
random.seed(gl_st.random_seed)

if gl_st.gpu_fraction == '':
    raise ValueError("--gpu_fraction should be defined")

frac = utils.calc_gpu_fraction(gl_st.gpu_fraction)
gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=frac)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:        
    if not tf.test.is_gpu_available() and gl_st.use_gpu:
        raise Exception("use_gpu flag is true when no GPUs are available")
    
    if ag_st.agent_type == 'dqn':
        
        agent = DQNAgent(cnf, environment, sess)
        
    elif ag_st.agent_type == 'hdqn':         
        agent = HDQNAgent(cnf, environment, sess)
        
    else:
        raise ValueError("Wrong agent %s".format())
        
    
    
    if ag_st.mode == 'train':
        agent.train()
    elif ag_st.mode == 'play':
        agent.play()
    elif ag_st.mode == 'graph':
        sys.exit(0)
    else:
        raise ValueError("Wrong mode " + str(gl.mode))
#if __name__ == '__main__':
#    tf.app.run()
