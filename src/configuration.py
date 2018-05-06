import os
import inspect
import sys
import glob
import utils
import math
import logging
from abc import ABCMeta, abstractmethod, abstractproperty
from pprint import pformat
from pynput.keyboard import Key
class Constants:
    

    key_to_sf = {
        'Key.up'     : 65362,
        'Key.right'  : 65363,
        'Key.down'   : 65364,
        'Key.left'  : 65361,
        'Key.space'  : 32,
        'Key.esc'    : -1,
        'wait'     : 0
    }
    
    SF_action_spaces = {
        'SFC-v0'   : ['Key.up', 'Key.right', 'Key.left', 'wait'],
        'SF-v0'    : [],
        'SFS-v0'   : [],
        'AIM-v0'   : []
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
        'SFC-v0'   : 5,
        'SF-v0'    : 0,
        'SFS-v0'   : 0,
        'AIM-v0'   : 0
            }
#    print(key_to_action)
#    print(action_to_sf)
    
    MDP_envs = ['stochastic_mdp-v0', 'ez_mdp-v0', 'trap_mdp-v0', 'key_mdp-v0']
    GYM_envs = ['CartPole-v0']
    env_names = SF_envs + MDP_envs + GYM_envs
    
    ### GOALS
    c = 2 * math.pi
    c34 = 3 / 4 * c
    c12 = 1 / 2 * c
    c14 = 1 / 4 * c
    
    #oneqpi = math.pi * 1 / 4
    
class Configuration:
    def __init__(self):
        self.gl = None      #Global settings
        self.ag = None      #Agent settings
        self.env = None     #Environment settings
        
    def set_agent_settings(self, settings):
        self.ag = settings
        
    def set_environment_settings(self, settings):
        self.env = settings
        
    def set_global_settings(self, settings):
        self.gl = settings
    
    def to_dict(self):
        dictionary = {}
        for settings in ['ag', 'env', 'gl']:
            try:
                dictionary[settings] = getattr(self, settings).to_dict()
            except Exception as e:
                print(e)
                
        return dictionary
    
    def to_str(self): return pformat(self.to_dict())
    
    def print(self):
        msg =  "\n" + self.to_str()
        logging.info(msg)   
        
    @property
    def model_dir(self):
        chain = []
        for attr_fullname in self.gl.attrs_in_dir:
            [attr_type, attr_name] = attr_fullname.split('.')
            try:
                attr_value = getattr(getattr(self, attr_type), attr_name)
            except AttributeError:
                attr_value = ''
            if 'architecture' in attr_name:
                value = '-'.join([str(l) for l in attr_value])
            else:
                value = str(attr_value)
            attr_name_initials = ''.join([word[0] for word in attr_name.split('_')])
            part = attr_name_initials + str(value)
            chain.append(part)
        result = '_'.join(chain)
        return result        
class GenericSettings():
        
    def update(self, new_attrs, add = False):
        """
        Add new attributes and overwrite existing ones.
        
        Args:
            new_attrs: dict ot settings object whose attributes will be added
            to the instance
        """
        if type(new_attrs) is dict:
            for key, value in new_attrs.items():
                
                if value is None or (not hasattr(self, key) and not add):
                    """When new_attrs is flags set through command line
                    parameters the default value is None so in those cases
                    we keep the instance value"""
                    continue
                else:
                    old_value = getattr(self, key) if hasattr(self, key) else '_'
                    logging.debug("Updated %s: %s -> %s", key, str(old_value),
                                                          str(value))
                    setattr(self, key, value)
                    
        elif isinstance(new_attrs, GenericSettings):
            raise NotImplementedError
        else:
            raise ValueError
                
    def to_dict(self): return vars(self).copy()
    def to_str(self): return pformat(self.to_dict())
        
    def print(self):
        msg =  "\n" + self.to_str()
        #print(msg)
        logging.info(msg)
            
    
    def to_disk(self, filepath):
        #TODO test this method
        content = self.to_str()
        with open(filepath) as fp:
            fp.write(content)

    
class GlobalSettings(GenericSettings):
    def __init__(self, new_attrs = {}):        
        self.display_prob = .01
        self.log_level = 'INFO'
        self.new_instance = True
        self.date = utils.get_timestamp()

        self.use_gpu = True
        self.gpu_fraction = '1/1'
        self.random_seed = 7
        self.root_dir = os.path.normpath(os.path.join(os.path.dirname(
                                        os.path.realpath(__file__)), ".."))
        self.environments_dir = os.path.join(self.root_dir, 'Environments')
    
        self.env_dirs = [
            os.path.join(self.root_dir, 'Environments','gym-stochastic-mdp'),
            os.path.join(self.root_dir, 'src','rainbow'),
            os.path.join(self.root_dir,  'Environments','gym-stochastic-mdp',
                                               'gym_stochastic_mdp','envs'),
            os.path.join(self.root_dir,  'Environments','SpaceFortress',
                                               'gym_space_fortress','envs'),
            os.path.join(self.root_dir,  'Environments','SpaceFortress',
                                               'gym_space_fortress', 'envs',
                                               'space_fortress'),
            os.path.join(self.root_dir,  'Environments','SpaceFortress')]
        #TODO clean path loadings
        self.ignore = ['display','new_instance','env_dirs','root_dir', 'ignore',
                       'use_gpu', 'gpu_fraction', 'is_train', 'prefix']
        self.attrs_in_dir = [
#                 'env.factor',
                 'gl.date',
                 'env.env_name',
#                 'ag.agent_type',
#                 'env.right_failure_prob', 
#                 'env.total_states',
                 'ag.architecture',
                 'ag.double_q',
                 'ag.dueling',
                 'ag.pmemory',
                 'ag.memory_size',
                 'gl.random_seed'
                 
#                 'ag.learning_rate_minimum',
#                 'ag.learning_rate',
#                 'ag.learning_rate_decay'
                 ]
        self.checkpoint_dir = '' #TODO
        self.logs_dir = os.path.join(self.root_dir, 'src', 'logs') #TODO
        self.settings_dir = '' #TODO
        self.randomize = False
        self.update(new_attrs)
        

class AgentSettings(GenericSettings):
    def __init__(self, scale = 1):
        self.scale = scale
        self.mode = 'train'
        self.pmemory = False
        self.max_step = self.scale * 5000
        self.double_q = False
        self.dueling = False
    
    def scale_attrs(self, attr_list):
        for attr in attr_list:
            normal = getattr(self, attr)
            scaled = self.scale * normal
            setattr(self, attr, scaled)
            
class HumanSettings(AgentSettings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = 'human'
        
        
class DQNSettings(AgentSettings):
    """
    Configuration of the DQN agent
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = 'dqn'
        #self.max_step = 500 * self.scale
        self.memory_size = 5000000
        
        self.batch_size = 32
        self.random_start = 30
        
        self.discount = 0.99
        self.target_q_update_step = 1 * self.scale
        self.learning_rate = 0.00025
        self.learning_rate_minimum = 0.00025
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = int(.001 * self.max_step)
        
        self.ep_end = 0.05
        self.ep_start = 1.
        self.ep_end_t = int(self.max_step * .75)
        
        self.history_length = 1
        self.train_frequency = 4
        #self.learn_start = 5. * self.scale
        
        self.architecture = [25, 25]
        self.architecture_duel = [16]
        
        self.test_step = 3000#int(self.max_step / 10)
        self.save_step = self.test_step * 10
        
        self.activation_fn = 'relu'
        self.prefix = ''
        
        
    
class hDQNSettings(AgentSettings):
    """
    Configuration
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = 'hdqn'
        self.architecture = [25, 25]
        self.architecture_duel = [10]
        self.mc = MetaControllerSettings(*args, **kwargs)
        self.c = ControllerSettings(*args, **kwargs)
        self.random_start = 30
        self.discount = 0.99
       
        
    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.mc.architecture = self.architecture
        self.c.architecture = self.architecture
        self.mc.architecture_duel = self.architecture_duel
        self.c.architecture_duel = self.architecture_duel
        
    def to_dict(self):       
        dictionary = vars(self).copy()
        dictionary['mc'] = self.mc.to_dict()
        dictionary['c'] = self.c.to_dict()
        return dictionary
        



class ControllerSettings(AgentSettings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.history_length = 1
        self.intrinsic_time_penalty = 0.01
        """
        Intrinsic rewards. If episode ends while a goal is being pursued (and
        haven't been accomplished yet), how should controller interpret that?
            1) Reward 0
            2) Don't learn transition. Only works if intrinsic
               time penalty is activated                                                                          
        """
        
        self.memory_size = 100000        
#        self.max_step = 500 * self.scale        
        self.batch_size = 32
        self.random_start = 30
        
        
        self.target_q_update_step = 1 * self.scale
        self.learning_rate = 0.0005
        self.learning_rate_minimum = 0.00025
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = int(.01 * self.max_step)
        
        self.ep_end = 0.05
        self.ep_start = 1.
        self.ep_end_t = int(self.max_step / 2)
        
        
        
        self.architecture = None
        self.architecture_duel = None
        
        self.test_step = 1000#min(5 * self.scale, 500)
        self.save_step = self.test_step * 10
        self.activation_fn = 'relu'
        
        self.ignore = ['ignore']
        self.prefix = 'c'
        
        self.train_frequency = 4
        #Visualize weights initialization in the histogram
        self.learn_start = 1000#min(5. * self.scale, self.test_step)
    
    
    
    
class MetaControllerSettings(AgentSettings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.history_length = 1    
        
        self.memory_size = 100000# * self.scale
         
        #max_step = 5000 * scale
        
        self.batch_size = 32
        self.random_start = 30
        
        
        self.target_q_update_step = 1 * self.scale
        self.learning_rate = 0.001
        self.learning_rate_minimum = 0.00025
        self.learning_rate_decay = 0.94
        self.learning_rate_decay_step = 5 * self.scale
        
        self.ep_end = 0.05
        self.ep_start = 1.
        self.ep_end_t = int(self.max_step / 2)
        
        self.train_frequency = 4
        self.learn_start = 1000#min(5. * self.scale, 20000)
        
        self.architecture = None
        self.architecture_duel = None
        
#        self.test_step = min(5 * self.scale, 500)
#        self.save_step = self.test_step * 10
        self.activation_fn = 'relu'
        
#        self.ignore = ['ignore']
        self.prefix = 'mc'
    
class EnvironmentSettings(GenericSettings):
    def __init__(self):
        self.env_name = ''   
        self.random_start = False 
        self.action_repeat = 1    
        self.right_failure_prob = 0.
        
class EZ_MDPSettings(EnvironmentSettings):
    def __init__(self, new_attrs):
        super().__init__()
        self.total_states = 6
        self.initial_state = 1
        self.terminal_states = [0, 5]
        self.total_actions = 2
        self.right_failure_prob = 0.
        self.update(new_attrs)
       
class Key_MDPSettings(EnvironmentSettings):
     def __init__(self, new_attrs): 
        super().__init__()
        self.factor = 3
        self.update(new_attrs)
        self.total_states = self.factor ** 2
        self.initial_state = int(self.total_states / 2)
        self.random_reset = True
        self.time_penalty = 0.
        
        
class Stochastic_MDPSettings(EnvironmentSettings):
    def __init__(self, new_attrs):
        super().__init__()
        self.total_states = 6
        self.initial_state = 1
        self.terminal_states = [0]
        self.total_actions = 2
        self.right_failure_prob = 0.5
        self.update(new_attrs)

           
        
          
class Trap_MDPSettings(EnvironmentSettings):
    def __init__(self, new_attrs):
        super().__init__()
        self.total_states = 6
        self.initial_state = 1
        self.terminal_states = [0, 5]
        self.total_actions = 2
        self.update(new_attrs)
     
        self.trap_states = [3, 4]
        

     
class RenderSpeed():
	# actually more of a render delay than speed 
	DEBUG=0
	SLOW=42
	MEDIUM=20
	FAST=8        
    
class SpaceFortressSettings(EnvironmentSettings):
    def __init__(self, new_attrs):
        super().__init__()
        self.no_direction = False
        self.library_path = os.path.join('..','Environments','SpaceFortress',
                                         'gym_space_fortress','envs',
                                         'space_fortress','shared')
        self.libsuffix = ""
        
#        self.screen_width = 84
#        self.screen_height = 84
#        self.render_mode = "idk" #minimal, rgb_array
        self.render_delay = 1

        self.record = False
        self.stats = False
        self.update(new_attrs)

class SpaceFortressControlSettings(SpaceFortressSettings):
    def __init__(self, new_attrs):
        super().__init__(new_attrs)
        self.time_penalty = 0.00
        



