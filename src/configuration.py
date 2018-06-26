import os
import pprint
import utils
"""
Class for hyperparameters and other configuration aspects.
[IMPORTANT] Parameters set through command line will overwrite whatever is here
"""

    
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
                pass#print(e)
                
        return dictionary
    
    def to_str(self): return pprint.pformat(self.to_dict())
     
    @property
    def model_name(self):
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
                    #old_value = getattr(self, key) if hasattr(self, key) else '_'
                    #logging.debug("Updated %s: %s -> %s", key, str(old_value),
                    #                                      str(value))
                    setattr(self, key, value)
                    
        elif isinstance(new_attrs, GenericSettings):
            raise NotImplementedError
        else:
            raise ValueError
                
    def to_dict(self): return vars(self).copy()
    def to_str(self): return pprint.pformat(self.to_dict())
        
    def print(self):
        msg =  "\n" + self.to_str()
        print(msg)
#        logging.info(msg)
            
    
    def to_disk(self, filepath):
        #TODO test this method
        content = self.to_str()
        with open(filepath) as fp:
            fp.write(content)

    
class GlobalSettings(GenericSettings):
    def __init__(self, new_attrs = {}):        
        self.display_prob = .0
        self.log_level = 'INFO'
        self.new_instance = True
        self.date = utils.get_timestamp()
        self.paralel = 0
        self.use_gpu = 0
        self.gpu_fraction = '1/1'
        self.random_seed = 7
        self.watch = 0
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
#                 'ag.mode',
                 'gl.date',
                 'ag.goal_group',
                 'env.env_name',
                 'env.ez',
                 'env.mines_activated',
                 'ag.agent_type',
#                 'env.right_failure_prob', 
#                 'env.total_states',
                 'ag.architecture',
                 'ag.double_q',
                 'ag.dueling',
                 'ag.pmemory',
                 #'ag.memory_size',
                 'env.action_repeat',
                 'gl.random_seed',
                 
                 
#                 'ag.learning_rate_minimum',
#                 'ag.learning_rate',
#                 'ag.learning_rate_decay'
                 ]
        self.others_dir = os.path.join(self.root_dir,  'Others')
        import platform
        if platform.linux_distribution()[0] == 'Ubuntu':
            #Ponyland server
            self.data_dir = '/vol/tensusers/vgarciacazorla/'
        else:
            #NLR server
            self.data_dir = self.others_dir
        self.checkpoints_dir = os.path.join(self.data_dir, 'checkpoints') #TODO
        self.logs_dir = os.path.join(self.data_dir, 'logs') #TODO
        self.randomize = False
        self.update(new_attrs)
        

class AgentSettings(GenericSettings):
    def __init__(self, scale = 1):
        self.scale = scale
        self.mode = 'train'
        self.pmemory = 0
        self.max_step = self.scale * 5000
        self.double_q = 0
        self.dueling = 0
        self.fresh_start = 0
        self.experiment_name = ''
    
    def scale_attrs(self, attr_list):
        for attr in attr_list:
            normal = getattr(self, attr)
            scaled = self.scale * normal
            setattr(self, attr, scaled)
            
class HumanSettings(AgentSettings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = 'human'
        self.goal_group = 1
        

class DQNSettings(AgentSettings):
    """
    Configuration of the DQN agent
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = 'dqn'
        self.memory_size = int(1e6)
        
        self.batch_size = 32
        self.random_start = 30
        
        self.discount = 0.99
        self.target_q_update_step = 1 * self.scale
        self.learning_rate = 5*1e-4
        self.learning_rate_minimum = 2*1e-4
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 5 * self.scale
        
        self.ep_end = 0.05
        self.ep_start = 1.
        self.ep_end_t_perc = .5
        
        self.history_length = 1
        self.train_frequency = 4
        self.learn_start = 10000
        
        self.architecture = [512, 512]
        self.architecture_duel = [128, 128]
        
        self.test_step = 10000#int(self.max_step / 10)
        self.save_step = self.test_step * 10
        
        self.activation_fn = 'relu'
        self.prefix = ''
        
        self.memory_minimum = 10000
        
        
    
class hDQNSettings(AgentSettings):
    """
    Configuration
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = 'hdqn'
        self.architecture = [512, 512]
        self.architecture_duel = [128, 128]
        self.memory_size = int(5e4)
        self.mc = MetaControllerSettings(*args, **kwargs)
        self.c = ControllerSettings(*args, **kwargs)
        self.random_start = 30
        self.discount = 0.99
        self.goal_group = 0
        self.save_step = 4       
        
    def update(self, args):
        super().update(args)
        self.mc.architecture = self.architecture
        self.c.architecture = self.architecture
        self.mc.architecture_duel = self.architecture_duel
        self.c.architecture_duel = self.architecture_duel
        if 'ep_start' in args:
            self.mc.update({'ep_start' : args['ep_start']})
       
        
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
        
        self.memory_size = int(1e6)
#        self.max_step = 500 * self.scale        
        self.batch_size = 32
        self.random_start = 30
        
        
        self.target_q_update_step = 1 * self.scale
        self.learning_rate = 5*1e-4
        self.learning_rate_minimum = 2*1e-4
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 5 * self.scale
        
       
        self.architecture = None
        self.architecture_duel = None
        
        self.test_step = 10000#min(5 * self.scale, 500)
        self.save_step = self.test_step * 10
        self.activation_fn = 'relu'
        
        self.ignore = ['ignore']
        self.prefix = 'c'
        
        self.train_frequency = 4
        #Visualize weights initialization in the histogram
        self.learn_start = 10000#min(5. * self.scale, self.test_step)
        self.learnt_threshold = 0.8
    
        self.memory_minimum = 10000
    
    
    
class MetaControllerSettings(AgentSettings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.history_length = 1    
        
        self.memory_size = int(5e4)# * self.scale
         
        #max_step = 5000 * scale
        
        self.batch_size = 32
        self.random_start = 30        
        
        self.target_q_update_step = 1 * self.scale
        self.learning_rate = 5*1e-4
        self.learning_rate_minimum = 2*1e-4
        self.learning_rate_decay = 0.94
        self.learning_rate_decay_step = 5 * self.scale
        
        self.ep_end = 0.05
        self.ep_start = 1.
        self.ep_end_t_perc = .5
        
        self.train_frequency = 4
        self.learn_start = 1000
        
        self.architecture = None
        self.architecture_duel = None
        

        self.activation_fn = 'relu'

        self.prefix = 'mc'
        self.memory_minimum = 1000
        
    
class EnvironmentSettings(GenericSettings):
    def __init__(self):
        self.env_name = ''   
        self.random_start = False 
        self.action_repeat = 5    
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
        self.random_reset = False
        self.time_penalty = 1e-2
        
        
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
        
   
    
class SpaceFortressSettings(EnvironmentSettings):
    def __init__(self, new_attrs):
        super().__init__()
        self.no_direction = False
        self.library_path = os.path.join('..','Environments','SpaceFortress',
                                         'gym_space_fortress','envs',
                                         'space_fortress','shared')
     
        self.render_delay = 10

   
        self.mines_activated = 1
       
        self.ship_lifes = 3
        self.fortress_lifes = 11
        self.max_loops = 2000 #Useful for stopping when the game crashes
        self.time_penalty = 0.01
        
        self.final_double_shot_reward = 1
        
        self.ez = 1
        
        
        
        
        self.min_steps_between_shots = 5
        self.min_steps_between_fortress_hits = 5
        self.max_steps_after_mine_appear = 40 # 2 seconds
        self.update(new_attrs)
    def set_reward_function(self):
        reward = 1 if self.ez else 1
        self.hit_fortress_reward = reward
        self.hit_mine_reward = reward
        
        self.fast_shooting_penalty = 1
        self.wrapping_penalty = 1
        self.hit_by_fortress_penalty = 1
        self.hit_by_mine_penalty = 1
        




