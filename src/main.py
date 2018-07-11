import os
import random
import argparse

import tensorflow as tf

from environment import Environment
from experiments import Experiment
import configuration
from constants import Constants as CT
import utils
from hDQN_agent import HDQNAgent
from DQN_agent import DQNAgent
try:
    from human_agent import HumanAgent
except Exception as e:
    print("Human agent not imported: %s" % (str(e)))
    
"""
Parameters defined here (command line) will overwrite those defined in
configuration.py
"""
parser = argparse.ArgumentParser()


# GLOBAL PARAMETERS
gl_args = parser.add_argument_group('Global')
gl_args.add_argument("--use_gpu", default = None, type = utils.str2bool)
gl_args.add_argument("--gpu_fraction", default = None)
gl_args.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
gl_args.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Log level.")
gl_args.add_argument("--display_prob", default = None, type = float)
gl_args.add_argument("--watch", default = None, type = utils.str2bool)
gl_args.add_argument("--parallel", default = 0, type = int)
gl_args.add_argument("--date", default = None, type = str)


# ENVIRONMENT PARAMETERS
env_args = parser.add_argument_group('Environment')
env_args.add_argument("--mdp_prob", default = None)
env_args.add_argument("--env_name", choices = CT.env_names ,default = "SF-v0")
env_args.add_argument("--right_failure_prob", default = None, type = float)
env_args.add_argument("--total_states", default = None, type = int)
env_args.add_argument("--factor", default = None, type = int)
env_args.add_argument("--render_delay", default = None, type = int)
env_args.add_argument("--action_repeat", default = None, type = int)
env_args.add_argument("--mines_activated", default = None, type = utils.str2bool)
env_args.add_argument("--ez", default = None, type = utils.str2bool)


# AGENT PARAMETERS
ag_args = parser.add_argument_group('Agent')
ag_args.add_argument("--scale", default = 50, type = int)
ag_args.add_argument("--agent_type", choices = ['dqn', 'hdqn', 'human', None], default = None, type = str)
ag_args.add_argument("--architecture", default = None, type = str)
ag_args.add_argument("--mode", default = 'train', type = str)
ag_args.add_argument("--learning_rate", default = None, type = float)
ag_args.add_argument("--learning_rate_minimum", default = None, type = float)
ag_args.add_argument("--learning_rate_decay", default = None, type = float)
ag_args.add_argument("--double_q", default = None, type = utils.str2bool)
ag_args.add_argument("--dueling", default = None, type = utils.str2bool)
ag_args.add_argument("--pmemory", default = None, type = utils.str2bool)
ag_args.add_argument("--c_architecture", default = None, type = str)
ag_args.add_argument("--mc_architecture", default = None, type = str)
ag_args.add_argument("--c_double_q", default = None, type = utils.str2bool)
ag_args.add_argument("--c_dueling", default = None, type = utils.str2bool)
ag_args.add_argument("--c_pmemory", default = None, type = utils.str2bool)
ag_args.add_argument("--mc_double_q", default = None, type = utils.str2bool)
ag_args.add_argument("--mc_dueling", default = None, type = utils.str2bool)
ag_args.add_argument("--mc_pmemory", default = None, type = utils.str2bool)
ag_args.add_argument("--memory_size", default = None, type = int)
ag_args.add_argument("--goal_group", default = None, type = int)
ag_args.add_argument("--ep_start", default = None, type = float)


def execute_experiment(args):
    #### HARD RULES
    if args['parallel'] != 0:
        args['use_gpu'] = 0
    if args['agent_type'] == 'human':
        args['use_gpu'] = 0
        #args['render_delay'] = 0
        args['mode'] = 'play'
        args['display_prob'] = 1
   
    if args['env_name'] == 'key_mdp-v0':
        args['action_repeat'] = 1
    arch_names = [n for n in args.keys() if 'architecture' in n]
    for arch_name in arch_names:
        if args[arch_name] is None:
            continue
        else:
            args[arch_name] = args[arch_name].split('-')
        
    cnf = configuration.Configuration()
    
    #Global settings
    gl_st = configuration.GlobalSettings(args)
   
    
    cnf.set_global_settings(gl_st)
    
    #Agent settings
    if args['agent_type'] == 'dqn':
        ag_st = configuration.DQNSettings(args['scale'])
    elif args['agent_type'] == 'hdqn':
        ag_st = configuration.hDQNSettings(args['scale'])
    elif args['agent_type'] == 'human':
        ag_st = configuration.HumanSettings()
    else:
        raise ValueError("Wrong agent %s" % args['agent_type'])
     

    ag_st.update(args)
    cnf.set_agent_settings(ag_st)
    
    #Environment settings
    utils.insert_dirs(cnf.gl.env_dirs)
      
    
    if args['env_name'] == 'SFC-v0':
        #Space Fortress
        env_st = configuration.SpaceFortressSettings(new_attrs = args)
        
    elif args['env_name'] == 'AIM-v0':
        #Space Fortress
        env_st = configuration.SpaceFortressSettings(new_attrs = args)
        
    elif args['env_name'] == 'SF-v0':
        #Space Fortress
        env_st = configuration.SpaceFortressSettings(new_attrs = args)
        env_st.set_reward_function()
        
    elif args['env_name'] == 'stochastic_mdp-v0':
        #MDP
        env_st = configuration.Stochastic_MDPSettings(new_attrs = args)
    elif args['env_name'] == 'ez_mdp-v0':
        #MDP
        env_st = configuration.EZ_MDPSettings(new_attrs = args)
    elif args['env_name'] == 'key_mdp-v0':
        #MDP
        env_st = configuration.Key_MDPSettings(new_attrs = args)
    elif args['env_name'] == 'trap_mdp-v0':
        #MDP
        env_st = configuration.Trap_MDPSettings(new_attrs = args)
    elif args['env_name'] == 'CartPole-v0':
        #MDP
        env_st = configuration.CartPoleSettings(new_attrs = args)
    else:
        raise ValueError("Wrong env_name %s, (env_names: s%)"\
                         .format(args['env_name'], ', '.join(CT.env_names)))
    
    cnf.set_environment_settings(env_st)
    environment = Environment(cnf)
    
    
    tf.set_random_seed(gl_st.random_seed)
    random.seed(gl_st.random_seed)
    
    if gl_st.gpu_fraction == '':
        raise ValueError("--gpu_fraction should be defined")
    
    if not gl_st.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
   
    frac = utils.calc_gpu_fraction(gl_st.gpu_fraction)
    gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=frac)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:        

        
        if ag_st.agent_type == 'dqn':
            agent = DQNAgent(cnf, environment, sess)
            
        elif ag_st.agent_type == 'hdqn':         
            agent = HDQNAgent(cnf, environment, sess)
            
        elif ag_st.agent_type == 'human':
            agent = HumanAgent(cnf, environment)
        else:
            raise ValueError("Wrong agent %s".format())
            
        
        
        if ag_st.mode == 'train':
            agent.train()
        elif ag_st.mode == 'play':
            agent.play()
        elif ag_st.mode == 'graph':
            pass
        else:
            raise ValueError("Wrong mode " + str(ag_st.mode))
        
        #agent.show_attrs()
    tf.reset_default_graph()


##############################################
##                   MAIN
##############################################
if __name__ == "__main__":
    # Parse arguments
    args = vars(parser.parse_args())
    
    
    if 'exp' in args['mode']:
        # Experiment mode. We will run more than one experiment (Experiments.py)
        exp_name = args['mode']    
        experiment = Experiment(exp_name, args['parallel'])
        args_list = experiment.get_args_list()
    else:
        # Not in experiment mode means that we are only running one experiment
        args_list = [args]
    
    if args['parallel'] == 0:
        # Execute experiments sequentially
        for args_ in args_list:
            execute_experiment(args_)
    else:
        # Execute experiments in parallel
        from multiprocessing import Pool    
        n_processes = args['parallel']    
        with Pool(n_processes) as pool:
            pool.starmap(execute_experiment, zip(args_list))
    print("Done :D")
        
