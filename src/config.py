class AgentConfig(object):
	scale = 500
	display = False


	max_step = 5000 * scale
	memory_size = 100 * scale

	batch_size = 32
	random_start = 30

	discount = 0.99
	target_q_update_step = 1 * scale
	learning_rate = 0.00025
	learning_rate_minimum = 0.00025
	learning_rate_decay = 0.96
	learning_rate_decay_step = 5 * scale

	ep_end = 0.1
	ep_start = 1.
	ep_end_t = memory_size

	history_length = 1
	train_frequency = 4
	learn_start = 5. * scale

	min_delta = -1
	max_delta = 1

	double_q = False
	dueling = False


	_test_step = 5 * scale
	_save_step = _test_step * 10

class EnvironmentConfig(object):
	#env_name = 'stochastic_mdp-v0'
	env_name = 'ez_mdp-v0'
	state_size = 6
	max_reward = 1.
	min_reward = 0

class DQNConfig(AgentConfig, EnvironmentConfig):
	model = ''
	pass

class M1(DQNConfig):
	backend = 'tf'
	env_type = 'detail'
	action_repeat = 1

def get_config(FLAGS):
	config = M1	
	for k, v in FLAGS.__dict__['__flags'].items():
		if hasattr(config, k):						 
			setattr(config, k, v)
		print(k,v)
	return config
