from __future__ import print_function
import logging
import random
import tensorflow as tf

from environment import Environment

import DQN
import hDQN
import configuration
from configuration import Constants as CT

import utils
import sys
import argparse
from pprint import pformat

parser = argparse.ArgumentParser()


# GLOBAL PARAMETERS
gl_args = parser.add_argument_group('Global')
gl_args.add_argument("--use_gpu", default = None, type = bool)
gl_args.add_argument("--gpu_fraction", default = None)
gl_args.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
gl_args.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Log level.")
gl_args.add_argument("--display_prob", default = None, type = float)

# ENVIRONMENT PARAMETERS
env_args = parser.add_argument_group('Environment')
env_args.add_argument("--mdp_prob", default = None)
env_args.add_argument("--env_name", default = "ez_mdp-v0")
env_args.add_argument("--right_failure_prob", default = None)
# AGENT PARAMETERS
ag_args = parser.add_argument_group('Agent')
ag_args.add_argument("--scale", default = 50, type = int)
ag_args.add_argument("--agent", default = None, type = str)
ag_args.add_argument("--train", default = None, type = bool)


#flags = tf.app.flags
#
## Model
#flags.DEFINE_string('agent', 'dqn', 'Which RL agent to use')
#
#flags.DEFINE_boolean('new_instance', None, 'Create new checkpoint and logs folder')
#
## Environment
#flags.DEFINE_string('env_name', None, 'The name of gym environment to use')
#flags.DEFINE_integer('action_repeat', None, 'The number of action to be repeated')
#
## Etc
#flags.DEFINE_boolean('use_gpu', None, 'Whether to use gpu or not')
#flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
##flags.DEFINE_boolean('display', None, 'Whether to do display the game screen or not')
#flags.DEFINE_string('mode', None, 'Whether to do training, testing or just seeing the graph')
#flags.DEFINE_integer('random_seed', None, 'Value of random seed')
##flags.DEFINE_integer('just_graph', None, 'Whether to just write the graph to TB or not')
#flags.DEFINE_integer('scale', 100, 'Scale to apply in configuration')
#flags.DEFINE_boolean('randomize', None, 'Whether to use a random agent or not')
#flags.DEFINE_float('display_episode_prob', None, 'Whether to use a random agent or not')

#
args = vars(parser.parse_args())


# Set random seed
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            level = getattr(logging, args['log_level']))

  
cnf = configuration.Configuration()

#Global settings
gl = configuration.GlobalSettings(args)

cnf.set_global_settings(gl)

#Agent settings
if args['agent'] == 'dqn':
    ag = configuration.DQNSettings(args['scale'])
elif args['agent'] == 'hdqn':
    ag = configuration.hDQNSettings(args['scale'])
else:
    raise ValueError("Wrong agent")
    
ag.update(args)
cnf.set_agent_settings(ag)

#Environment settings
utils.insert_dirs(cnf.gl.env_dirs)


if args['env_name'] in CT.SF_envs:
    #Space Fortress
    env = configuration.SpaceFortressSettings(new_attrs = args)
    
elif args['env_name'] in CT.MDP_envs:
    #MDP
    env = configuration.MDPSettings(new_attrs = args)
else:
    raise ValueError("Wrong env_name %s".format(args['env_name']))
env.print()
cnf.set_environment_settings(env)
env = Environment(cnf)
#config.update(env.configuration_attrs)
cnf.print()
sys.exit(0)


tf.set_random_seed(gl_config.random_seed)
random.seed(gl_config.random_seed)

if config.gpu_fraction == '':
    raise ValueError("--gpu_fraction should be defined")

frac = utils.calc_gpu_fraction(config.gpu_fraction)
gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=frac)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        

    if not tf.test.is_gpu_available() and config.use_gpu:
        raise Exception("use_gpu flag is true when no GPUs are available")
    
    if config.agent == 'dqn':
        agent = DQN.Agent(config, env, sess)
    elif config.agent == 'hdqn':         
        agent = hDQN.Agent(config, env, sess)
        
    
    
    if config.mode == 'train':
        agent.train()
    elif config.mode == 'play':
        agent.play()
    elif config.mode == 'graph':
        sys.exit(0)
    else:
        raise ValueError("Wrong mode " + str(config.mode))
#if __name__ == '__main__':
#    tf.app.run()
