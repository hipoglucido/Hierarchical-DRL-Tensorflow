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
            strings = attr_fullname.split('.')
            if len(strings) == 2:
                [attr_type, attr_name] = strings
                module = ''
                #obj = getattr(self, attr_type)
                f = lambda x: getattr(x, attr_type)
            else:
                [attr_type, module, attr_name] = strings
                f = lambda x: getattr(getattr(x, attr_type), module)
                #obj = getattr(getattr(self, attr_type), module)
                
            try:
                attr_value = getattr(f(self), attr_name)
            except AttributeError:
                attr_value = ''
            if 'architecture' in attr_name:
                value = '-'.join([str(l) for l in attr_value])
            else:
                value = str(attr_value)
            attr_name_initials = ''.join([word[0] for word in attr_name.split('_')])
            part = module.upper() + attr_name_initials + str(value)
            if value == '':
                continue
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
#                    old_value = getattr(self, key) if hasattr(self, key) else '_'
#                    print("Updated %s: %s -> %s" % (key, str(old_value),
#                                                          str(value)))
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
        self.date = utils.get_timestamp()
        self.parallel = 0
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

        self.attrs_in_dir = [
#                 'env.factor',
#                 'ag.mode',
                 'gl.date',
                 'ag.goal_group',
                 'env.env_name',
                 'env.sparse_rewards',
                 'env.reward_type',
                 #'env.mines_activated',
                 'ag.agent_type',
#                 'env.right_failure_prob', 
#                 'env.total_states',
                 'ag.architecture',
                 'ag.double_q',
                 'ag.dueling',
                 'ag.pmemory',
                 
                 
                 'ag.mc.architecture',
                 'ag.mc.double_q',
                 'ag.mc.dueling',
                 'ag.mc.pmemory',
                 
                 'ag.c.architecture',
                 'ag.c.double_q',
                 'ag.c.dueling',
                 'ag.c.pmemory',
                 #'ag.c.intrinsic_time_penalty',
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
        self.max_step = self.scale * 5000
        
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
        self.goal_group = 4
        

class DQNSettings(AgentSettings):
    """
    Configuration of the DQN agent
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = 'dqn'
        self.memory_size = int(1e6)
        
        self.batch_size = 32
        
        self.discount = 0.99
        self.target_q_update_step = 1 * self.scale
        self.learning_rate = 5*1e-4
        self.learning_rate_minimum = 2*1e-4
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 5 * self.scale
        
        self.ep_end = 0.05
        self.ep_start = 1.
        self.ep_end_t_perc = .8
        
        self.history_length = 1
        self.train_frequency = 4
        self.learn_start = 10000
        
        self.architecture = [512, 512]
        self.architecture_duel = [128]
        
        self.test_step = 10000
        
        self.activation_fn = 'relu'
        self.prefix = ''
        
        self.memory_minimum = 10000
        
        self.dueling = 1
        self.double_q = 1
        self.pmemory = 1
        
        
    
class hDQNSettings(AgentSettings):
    """
    Configuration
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = 'hdqn'
        
        
        self.mc = MetaControllerSettings(*args, **kwargs)
        self.c = ControllerSettings(*args, **kwargs)
        
        self.goal_group = 1 
        
        if 'ep_start' in args:
            self.mc.update({'ep_start' : args['ep_start']})
               
    def to_dict(self):       
        dictionary = vars(self).copy()
        dictionary['mc'] = self.mc.to_dict()
        dictionary['c'] = self.c.to_dict()
        return dictionary
        
    def update(self, args):
        super().update(args)
        for ag_name in ['c', 'mc']:
            ag = getattr(self, ag_name)
            args_copy = {}
            for k, v in args.items():
                new_key = k.replace("%s_" % ag_name, "", 1)
                args_copy[new_key] = v
            ag.update(args_copy)
        
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
        self.batch_size = 32
        
        self.target_q_update_step = 1 * self.scale
        self.learning_rate = 5*1e-4
        self.learning_rate_minimum = 2*1e-4
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 5 * self.scale
        self.discount = 0.99
       
        self.architecture = [512, 512]
        self.architecture_duel = [128]
        self.dueling = 1
        self.double_q = 1
        self.pmemory = 1
        
        self.test_step = 10000
        self.activation_fn = 'relu'
        
        
        self.train_frequency = 4
        #Visualize weights initialization in the histogram
        self.learn_start = 10000
        # C needs to reach `learnt_threshold` of its goals so that MC starts learning
        self.learnt_threshold = 0.95
        # In order to compute the rate of success of each goal, take only into
        # account the X last attempts
        self.goal_attempts_list_len = 1000
    
        self.memory_minimum = 10000
        

    
class MetaControllerSettings(AgentSettings):
        

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.history_length = 1    
        
        self.memory_size = int(5e4)
         
     
        
        self.batch_size = 32    
        
        self.target_q_update_step = 1 * self.scale
        self.learning_rate = 5*1e-4
        self.learning_rate_minimum = 2*1e-4
        self.learning_rate_decay = 0.94
        self.learning_rate_decay_step = 5 * self.scale
        self.discount = 0.99
        self.ep_end = 0.05
        self.ep_start = 1.
        self.ep_end_t_perc = .8
        
        self.train_frequency = 4
        self.learn_start = 1000
        
        self.architecture = [512, 512]
        self.architecture_duel = [128]
        self.dueling = 1
        self.double_q = 1
        self.pmemory = 1
        

        self.activation_fn = 'relu'

        self.prefix = 'mc'
        self.memory_minimum = 1000
        
    
class EnvironmentSettings(GenericSettings):
    def __init__(self):
        self.env_name = ''   
        self.random_start = False
        self.action_repeat = 1  
        self.right_failure_prob = 0.
        

       
class Key_MDPSettings(EnvironmentSettings):
     def __init__(self, new_attrs): 
        super().__init__()
        self.factor = 3
        self.reward_type = 1
        self.update(new_attrs)
        self.total_states = self.factor ** 2
        self.initial_state = int(self.total_states / 2)
        self.random_reset = False

     def set_reward_function(self):
        if self.reward_type == 1:
            self.time_penalty = 0
            self.small_reward = 0.1
            self.big_reward = 1
        elif self.reward_type == 2:
            self.time_penalty = .01
            self.small_reward = .1
            self.big_reward = 1
        else:
            assert 0, "Wrong reward type %s" % str(self.reward_type)
            

   
    
class SpaceFortressSettings(EnvironmentSettings):
    def __init__(self, new_attrs):
        super().__init__()
        self.no_direction = False
        self.library_path = os.path.join('..','Environments','SpaceFortress',
                                         'gym_space_fortress','envs',
                                         'space_fortress','shared')             
        self.ez = 0    # easy mode
        self.sparse_rewards = 1
        self.mines_activated = 1  
        self.render_delay = 10        
        self.ship_lifes = 3
        self.fortress_lifes = 11
        self.max_loops = 3000 #Useful for stopping when the game crashes  
        self.min_steps_between_shots = 5
        self.min_steps_between_fortress_hits = 5
        self.max_steps_after_mine_appear = 40 # 2 seconds
        self.update(new_attrs)
        
    def set_reward_function(self):
       
        sr = self.sparse_rewards
        # Positive rewards
        self.hit_fortress_reward = 0 if sr else 1
        self.hit_mine_reward = 0
        self.final_double_shot_reward = 1 if sr else 5      
        self.fast_shooting_penalty = 0 if sr else 5
        
        # Negative rewards
        self.wrapping_penalty = 1
        self.hit_by_fortress_penalty = 0 if sr else 1
        self.hit_by_mine_penalty = 0 if sr else 1
        self.time_penalty = 0.01        
        




