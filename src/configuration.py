import os
import inspect
import sys
import glob
import utils
import pprint

class Configuration():
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

class GlobalConfiguration(Configuration):
	
	env_name = 'trap_mdp-v0'
	display = False
	new_instance = True
	date = utils.get_timestamp()
	action_repeat = 1
	use_gpu = True
	gpu_fraction = '1/1'
	mode = 'train'
	random_seed = 7
	root_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
	envs_dirs = [os.path.join(root_dir, '..', 'Environments','gym-stochastic-mdp'),
				  os.path.join(root_dir, '..', 'Environments','gym-stochastic-mdp',
								  'gym_stochastic_mdp','envs')]
	
	ignore = ['display','new_instance','envs_dirs','root_dir', 'ignore',
		   'use_gpu', 'gpu_fraction', 'is_train', 'prefix']
	


class ControllerParameters(Configuration):
	scale = 500
	
	history_length = 4
	
	memory_size = 100 * scale
		
	max_step = 500 * scale

	batch_size = 8
	random_start = 30

	discount = 0.99
	target_q_update_step = 1 * scale
	learning_rate = 0.001
	learning_rate_minimum = 0.00025
	learning_rate_decay = 0.94
	learning_rate_decay_step = 5 * scale

	ep_end = 0.1
	ep_start = 1.
	ep_end_t = memory_size

	train_frequency = 4
	learn_start = 5. * scale

	architecture = [200, 300, 100, 50]
	_test_step = 5 * scale
	_save_step = _test_step * 10
	activation_fn = 'relu'
	
	ignore = ['ignore']
	prefix = 'c'
	
	
	
	
class MetaControllerParameters(Configuration):
	scale = 500
	
	history_length = 4	
	
	memory_size = 100 * scale
		
	#max_step = 5000 * scale

	batch_size = 64
	random_start = 30

	discount = 0.99
	target_q_update_step = 1 * scale
	learning_rate = 0.001
	learning_rate_minimum = 0.00025
	learning_rate_decay = 0.94
	learning_rate_decay_step = 5 * scale

	ep_end = 0.1
	ep_start = 1.
	ep_end_t = memory_size

	train_frequency = 4
	learn_start = 5. * scale

	architecture = [50, 75, 100, 50, 25, 10]
	
	_test_step = 5 * scale
	_save_step = _test_step * 10
	activation_fn = 'relu'
	
	ignore = ['ignore']
	prefix = 'mc'
	

	
class hDQNConfiguration(GlobalConfiguration):
	mc_params = MetaControllerParameters()
	c_params = ControllerParameters()
	random_start = 30
	
	
class DQNConfiguration(GlobalConfiguration):
	scale = 500

	max_step = 100 * scale
	memory_size = 5 * scale

	batch_size = 32
	random_start = 30

	discount = 0.99
	target_q_update_step = 1 * scale
	learning_rate = 0.001
	learning_rate_minimum = 0.00025
	learning_rate_decay = 0.93
	learning_rate_decay_step = 5 * scale

	ep_end = 0.1
	ep_start = 1.
	ep_end_t = memory_size

	history_length = 4
	train_frequency = 4
	learn_start = 5. * scale

	architecture = [500, 500, 500]


	_test_step = 5 * scale
	_save_step = _test_step * 10
	
	activation_fn = 'relu'
	prefix = ''
	




