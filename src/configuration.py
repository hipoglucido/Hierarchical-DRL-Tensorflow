import os
import inspect
import sys
import glob
import utils
import pprint

class GenericSettings():
    def __init__(self, scale):
        self.scale = scale
    def update(self, new_attrs):
        for key, value in new_attrs.items():
            
            if value is None:
                continue
            else:
                setattr(self, key, value)
                
    def insert_envs_paths(self):
        for env_dir in self.envs_dirs:
            sys.path.insert(0, env_dir)
            print("Added", env_dir)
            
    def as_list(self, ignore = True):
        ignore = self.ignore if ignore else []
        def aux(k, v, p): return "%s%s-%s" % (p, k, ",".join([str(i) for i in v])
                                                    if type(v) == list else v)
        if not self.new_instance:
            self.ignore.append('date')
        parts = [self.env_name]
        for k, v in inspect.getmembers(self):
            if isinstance(v, ControllerParameters):
                for k_, v_ in inspect.getmembers(v):
                    if k_.startswith("__"):
                        continue
                    parts = parts + [aux(k_, v_, 'C-')] if k_ not in v.ignore else parts                    
            elif isinstance(v, MetaControllerParameters):
                for k_, v_ in inspect.getmembers(v):
                    if k_.startswith("__"):
                        continue
                    parts = parts + [aux(k_, v_, 'MC-')] if k_ not in v.ignore else parts                    
            elif callable(v) or k in ignore or k.startswith('__'):
                continue
            else:
                parts.append(aux(k, v, ''))        
        return parts
    def print(self):
        elements = self.as_list(ignore = False)
        elements = [e for e in elements if e is not None]
        out = 'Configuration:\n' + '\n\t'.join(elements)
        print(out)


class SpaceFortressSettigns(GenericSettings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.no_direction = False
        self.libsuffix = ""
        
        self.screen_width = 84
        self.screen_height = 84
        
class MDPSettings(GenericSettings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_states = 6
        self.final_states = [0, 6]
    
class Configuration():
    def __init__(self):
        pass
    def set_agent_configuration(self, configuration):
        self.ag = configuration
        
    def set_environment_configuration(self, configuration):
        self.env = configuration
        
    def set_global_configuration(self, configuration):
        self.gl = configuration
    
class GlobalSettings(GenericSettings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.env_name = 'trap_mdp-v0'
        self.display = False
        self.display_episode_prob = 1.
        self.new_instance = True
        self.date = utils.get_timestamp()
        self.action_repeat = 1
        self.use_gpu = True
        self.gpu_fraction = '1/1'
        self.mode = 'train'
        self.random_seed = 7
        self.root_dir = os.path.normpath(os.path.join(os.path.dirname(
                                        os.path.realpath(__file__)), ".."))
        self.envs_dirs = [
            os.path.join(self.root_dir, '..', 'Environments','gym-stochastic-mdp'),
            os.path.join(self.root_dir, '..', 'Environments','gym-stochastic-mdp',
                                               'gym_stochastic_mdp','envs'),
            os.path.join(self.root_dir, '..', 'Environments','SpaceFortress',
                                               'gym','envs'),
            os.path.join(self.root_dir, '..', 'Environments','SpaceFortress',
                                               'gym')]
        
        self.ignore = ['display','new_instance','envs_dirs','root_dir', 'ignore',
                       'use_gpu', 'gpu_fraction', 'is_train', 'prefix']
        self.randomize = False


class ControllerParameters(GenericSettings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.history_length = 1
        
        self.memory_size = 100 * self.scale
        
        self.max_step = 500 * self.scale
        
        self.batch_size = 32
        self.random_start = 30
        
        self.discount = 0.99
        self.target_q_update_step = 1 * self.scale
        self.learning_rate = 0.001
        self.learning_rate_minimum = 0.00025
        self.learning_rate_decay = 0.94
        self.learning_rate_decay_step = 5 * self.scale
        
        self.ep_end = 0.1
        self.ep_start = 1.
        self.ep_end_t = self.memory_size
        
        self.train_frequency = 4
        self.learn_start = 5. * self.scale
        
        self.architecture = [500, 500, 500, 500, 500, 500, 500, 500]
        self.test_step = 5 * self.scale
        self.save_step = self.test_step * 10
        self.activation_fn = 'relu'
        
        self.ignore = ['ignore']
        self.prefix = 'c'
    
    
    
    
class MetaControllerParameters(GenericSettings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.history_length = 1    
        
        self.memory_size = 100 * self.scale
         
        #max_step = 5000 * scale
        
        self.batch_size = 64
        self.random_start = 30
        
        self.discount = 0.99
        self.target_q_update_step = 1 * self.scale
        self.learning_rate = 0.001
        self.learning_rate_minimum = 0.00025
        self.learning_rate_decay = 0.94
        self.learning_rate_decay_step = 5 * self.scale
        
        self.ep_end = 0.1
        self.ep_start = 1.
        self.ep_end_t = self.memory_size
        
        self.train_frequency = 4
        self.learn_start = 5. * self.scale
        
        self.architecture = [500, 500, 500]
        
        self.test_step = 5 * self.scale
        self.save_step = self.test_step * 10
        self.activation_fn = 'relu'
        
        self.ignore = ['ignore']
        self.prefix = 'mc'
    

    
class hDQNConfiguration(GenericSettings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc_params = MetaControllerParameters(*args, **kwargs)
        self.c_params = ControllerParameters(*args, **kwargs)
        self.random_start = 30
        
    
class DQNConfiguration(GenericSettings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.max_step = 100 * self.scale
        self.memory_size = 5 * self.scale
        
        self.batch_size = 32
        self.random_start = 30
        
        self.discount = 0.99
        self.target_q_update_step = 1 * self.scale
        self.learning_rate = 0.001
        self.learning_rate_minimum = 0.00025
        self.learning_rate_decay = 0.93
        self.learning_rate_decay_step = 5 * self.scale
        
        self.ep_end = 0.1
        self.ep_start = 1.
        self.ep_end_t = self.memory_size
        
        self.history_length = 4
        self.train_frequency = 4
        self.learn_start = 5. * self.scale
        
        self.architecture = [500, 500, 500]
        
        
        self.test_step = 5 * self.scale
        self.save_step = self.test_step * 10
        
        self.activation_fn = 'relu'
        self.prefix = ''
        








