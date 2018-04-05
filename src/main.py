from __future__ import print_function
import random
import tensorflow as tf

from environment import Environment
#from environment2 import GymEnvironment
import DQN
import hDQN
import configuration

import utils
import sys
#sys.path.insert(0, '/home/victorgarcia/work/Environments/gym-stochastic-mdp/gym_stochastic_mdp/envs/')


flags = tf.app.flags

# Model
flags.DEFINE_string('agent', 'dqn', 'Which RL agent to use')

flags.DEFINE_boolean('new_instance', None, 'Create new checkpoint and logs folder')

# Environment
flags.DEFINE_string('env_name', None, 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', None, 'The number of action to be repeated')

# Etc
flags.DEFINE_boolean('use_gpu', None, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
#flags.DEFINE_boolean('display', None, 'Whether to do display the game screen or not')
flags.DEFINE_string('mode', None, 'Whether to do training, testing or just seeing the graph')
flags.DEFINE_integer('random_seed', None, 'Value of random seed')
#flags.DEFINE_integer('just_graph', None, 'Whether to just write the graph to TB or not')
flags.DEFINE_integer('scale', 100, 'Scale to apply in configuration')
flags.DEFINE_boolean('randomize', None, 'Whether to use a random agent or not')
flags.DEFINE_float('display_episode_prob', None, 'Whether to use a random agent or not')

#


# Set random seed




def calc_gpu_fraction(fraction_string):
    idx, num = fraction_string.split('/')
    idx, num = float(idx), float(num)

    fraction = 1 / (num - idx + 1)
    print(" [*] GPU : %.4f" % fraction)
    return fraction



def main(_):
    try:
        flags_dict = flags.FLAGS.flag_values_dict()
    except AttributeError:
        flags_dict = flags.FLAGS.__dict__['__flags']
        
    gl = configuration.GlobalSettings()
    
    if flags.FLAGS.agent == 'dqn':
        ag_config = configuration.DQNConfiguration(flags_dict['scale'])
    elif flags.FLAGS.agent == 'hdqn':
        ag_config = configuration.hDQNConfiguration(flags_dict['scale'])
    else:
        raise ValueError("Wrong agent")
    ag_config.update(flags_dict)
    ag_config.print()
    ag_config.insert_envs_paths()
    sf_envs = ['SFC-v0']
    if flags.FLAGS.env_name in sf_envs:
        #Space Fortress
        env_config = configuration.SFConfig
        
    else:
        #MDP
        env_config = ag_config
    env = Environment(env_config, ag_config)
    #config.update(env.configuration_attrs)
    ag_config.print()
    
    
    
    tf.set_random_seed(ag_config.random_seed)
    random.seed(ag_config.random_seed)
    
    if config.gpu_fraction == '':
        raise ValueError("--gpu_fraction should be defined")
        
    gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=calc_gpu_fraction(config.gpu_fraction))
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
if __name__ == '__main__':
    tf.app.run()
