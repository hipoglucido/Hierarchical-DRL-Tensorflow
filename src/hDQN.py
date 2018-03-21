from __future__ import print_function
import os
import time
import random
import numpy as np
from tqdm import tqdm
from functools import reduce
import tensorflow as tf
import sys


from base import BaseModel
from history import History
from replay_memory import ReplayMemory
from ops import linear, clipped_error
from utils import get_time, save_pkl, load_pkl

class Agent(BaseModel):
	def __init__(self, config, environment, sess):
		super(Agent, self).__init__(config)
		self.sess = sess
		self.weight_dir = 'weights'

		self.env = environment
		self.mc_history = History(self.config.mc_params)
		self.c_history = History(self.config.c_params)
		
		self.mc_memory = ReplayMemory(self.config.mc_params, self.model_dir)
		self.c_memory = ReplayMemory(self.config.c_params, self.model_dir)
		
		
		with tf.variable_scope('c_step'):
			self.c_step_op = tf.Variable(0, trainable=False, name='c_step')
			self.c_step_input = tf.placeholder('int32', None,
												  name='c_step_input')
			self.c_step_assign_op = self.c_step_op.assign(self.c_step_input)
			
		with tf.variable_scope('mc_step'):
			self.mc_step_op = tf.Variable(0, trainable=False,
										    name='mc_step')
			self.mc_step_input = tf.placeholder('int32', None,
										   name='mc_step_input')
			self.mc_step_assign_op = self.mc_step_op.assign(self.mc_step_input)
				
		
		
		self.set_interfaces_lengths(config)
		self.build_hdqn(config)

	def predict_goal(self, s_t. test_ep = None):
	    ep = test_ep or (self.ep_end +
	        max(0., (self.ep_start - self.ep_end)
	          * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))
	
	    if random.random() < ep:
	      action = random.randrange(self.env.action_size)
	    else:
	      action = self.q_action.eval({self.s_t: [s_t]})[0]
	
	    return action		
		
	def train(self, config):
		
		mc_cnf = config.mc_params
		c_cnf = config.c_params
		
		mc_start_step = self.mc_step_op.eval()	
		c_start_step = self.c_step_op.eval()		
		
		num_game, self.update_count, ep_reward = 0, 0, 0.
		total_reward, self.total_loss, self.total_q = 0., 0., 0.
		max_avg_ep_reward = 0
		ep_rewards, actions, goals = [], [], []

		#screen, reward, action, terminal = self.env.new_random_game()
		screen, _, _, _ = self.env.new_game(False)
		for _ in range(self.mc_history.length):
			self.mc_history.add(screen)
		t_0 = time.time()
#		for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
		
		
		
		for self.mc_step in range(mc_start_step, c_cnf.max_step):
			if self.mc_step == c_cnf.learn_start:				
				num_game, self.update_count, ep_reward = 0, 0, 0.
				total_reward, self.total_loss, self.total_q = 0., 0., 0.
				ep_rewards, actions, goals = [], [], []
			
			# 1. predict
			aux = self.history.get()
			action = self.predict(aux)	
			
			# 2. act			
			screen, reward, terminal = self.env.act(action, is_training = True)

			# 3. observe
			self.observe(screen, reward, action, terminal)
			ep_reward += reward
			if terminal:
				#print("Terminal")
				screen, _, _, _ = self.env.new_game(False)
				
				for _ in range(self.history_length):
					self.history.add(screen)
				num_game += 1
				ep_rewards.append(ep_reward)
				ep_reward = 0.			
			
			actions.append(action)
			total_reward += reward

			if self.step >= self.learn_start:
				
				if self.step % self.test_step == self.test_step - 1:
					avg_reward = total_reward / self.test_step
					avg_loss = self.total_loss / self.update_count
					avg_q = self.total_q / self.update_count
					t_1 = time.time()
					time_, t_0 = t_1 - t_0, t_1
					
					try:
						max_ep_reward = np.max(ep_rewards)
						min_ep_reward = np.min(ep_rewards)
						avg_ep_reward = np.mean(ep_rewards)
					except Exception as e:
						print(str(e))
						max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0
					msg = ("\navg_r: {:.4f}, avg_l: {:.6f}, avg_q: {:.3f}, "+\
							"avg_ep_r: {:.2f}, max_ep_r: {:.2f}, min_ep_r: "+\
							"{:.2f}, secs: {:.1f}, #g: {}").format(
									avg_reward, avg_loss, avg_q,
									avg_ep_reward, max_ep_reward,
									min_ep_reward, time_, num_game)
					print(msg)
	
					if max_avg_ep_reward * 0.9 <= avg_ep_reward:
						self.step_assign_op.eval(
								{self.step_input: self.step + 1})
						self.save_model(self.step + 1)
	
						max_avg_ep_reward = max(max_avg_ep_reward,
											    avg_ep_reward)
	
					if self.step > 10:
						
						self.inject_summary({
								'average.reward': avg_reward,
								'average.loss': avg_loss,
								'average.q': avg_q,
								'time': time_,
								'episode.max reward': max_ep_reward,
								'episode.min reward': min_ep_reward,
								'episode.avg reward': avg_ep_reward,
								'num of game': num_game,
								'episode.rewards': ep_rewards,
								'actions': actions,
								'training.learning_rate': \
									self.learning_rate_op.eval(
										{self.learning_rate_step: self.step}),
							}, self.step)
	
					num_game = 0
					total_reward = 0.
					self.total_loss = 0.
					self.total_q = 0.
					self.update_count = 0
					ep_reward = 0.
					ep_rewards = []
					actions = []		

	def set_interfaces_lengths(self, config):	
		config.mc_params.q_input_length = self.env.state_size
		config.mc_params.q_output_length = self.env.goal_size
		config.c_params.q_input_length = self.env.state_size + \
														self.env.goal_size
		config.c_params.q_output_length = self.env.action_size
		
	def build_meta_controller(self, config):
		self.mc_w = {}
		self.mc_target_w = {}
		# training meta-controller network
		with tf.variable_scope('mc_prediction'):
			# tf Graph input
			self.mc_s_t = tf.placeholder("float",
					    [None, self.mc_history.length, self.state_size],
						name='mc_s_t')
			shape = self.mc_s_t.get_shape().as_list()
			self.mc_s_t_flat = tf.reshape(self.mc_s_t, [-1, reduce(
											lambda x, y: x * y, shape[1:])])			
			
			last_layer = self.mc_s_t_flat
			last_layer = self.add_dense_layers(config = config.mc_params,
											   input_layer = last_layer,
											   prefix = 'mc')
			self.mc_q, self.mc_w['q_w'], self.mc_w['q_b'] = linear(last_layer,
												  self.env.goal_size,
												  name='mc_q')
			print(self.mc_q)
			self.mc_q_goal= tf.argmax(self.mc_q, axis=1)
			
			q_summary = []
			avg_q = tf.reduce_mean(self.mc_q, 0)
			

			for idx in range(self.env.goal_size):
				q_summary.append(tf.summary.histogram('mc_q/%s' % idx, avg_q[idx]))
			self.q_summary = tf.summary.merge(q_summary, 'mc_q_summary')

		# target network
		self.create_target(config = config.mc_params)

		#Meta Controller optimizer
		with tf.variable_scope('mc_optimizer'):
			self.mc_target_q_t = tf.placeholder('float32', [None],
											   name='mc_target_q_t')
			self.mc_action = tf.placeholder('int64', [None], name='mc_action')

			mc_action_one_hot = tf.one_hot(self.mc_action, self.env.goal_size,
									   1.0, 0.0, name = 'mc_action_one_hot')
			mc_q_acted = tf.reduce_sum(self.mc_q * mc_action_one_hot,
								   reduction_indices = 1, name = 'mc_q_acted')

			mc_delta = self.mc_target_q_t - mc_q_acted

			#self.global_step = tf.Variable(0, trainable=False)

			self.mc_loss = tf.reduce_mean(clipped_error(mc_delta),
										 name = 'mc_loss')
			self.mc_learning_rate_step = tf.placeholder('int64', None,
											name='mc_learning_rate_step')
			self.mc_learning_rate_op = tf.maximum(
					config.mc_params.learning_rate_minimum,
					tf.train.exponential_decay(
						learning_rate = config.mc_params.learning_rate,
						global_step   = self.mc_learning_rate_step,
						decay_steps   = config.mc_params.learning_rate_decay_step,
						decay_rate    = config.mc_params.learning_rate_decay,
						staircase     = True))
			self.mc_optim = tf.train.RMSPropOptimizer(
								learning_rate = self.mc_learning_rate_op,
								momentum      = 0.95,
								epsilon       = 0.01).minimize(self.mc_loss)

	def build_controller(self, config):
		self.c_w = {}
		self.c_target_w = {}
	
		with tf.variable_scope('c_prediction'):
			input_size = self.env.state_size + self.env.goal_size
			self.c_s_t = tf.placeholder("float",
								[None, self.c_history.length, input_size],
								name = 'c_s_t')
			shape = self.c_s_t.get_shape().as_list()
			self.c_s_t_flat = tf.reshape(self.c_s_t, [-1, reduce(
					lambda x, y: x * y, shape[1:])])
			last_layer = self.c_s_t_flat
			last_layer = self.add_dense_layers(config = config.c_params,
											   input_layer = last_layer,
											   prefix = 'c')
			self.c_q, self.c_w['q_w'], self.c_w['q_b'] = linear(last_layer,
												  self.env.action_size,
												  name='c_q')
			print(self.c_q)
			self.c_q_action= tf.argmax(self.c_q, axis=1)
			
			q_summary = []
			avg_q = tf.reduce_mean(self.c_q, 0)
			

			for idx in range(self.env.action_size):
				q_summary.append(tf.summary.histogram('c_q/%s' % idx, avg_q[idx]))
			self.q_summary = tf.summary.merge(q_summary, 'c_q_summary')

		# target network
		self.create_target(config.c_params)
		
		
		#Controller optimizer
		with tf.variable_scope('c_optimizer'):
			self.c_target_q_t = tf.placeholder('float32', [None],
											   name='c_target_q_t')
			self.c_action = tf.placeholder('int64', [None], name='c_action')

			c_action_one_hot = tf.one_hot(self.c_action, self.env.action_size,
									   1.0, 0.0, name = 'c_action_one_hot')
			c_q_acted = tf.reduce_sum(self.c_q * c_action_one_hot,
								   reduction_indices = 1, name = 'c_q_acted')

			c_delta = self.c_target_q_t - c_q_acted

			#self.global_step = tf.Variable(0, trainable=False)

			self.c_loss = tf.reduce_mean(clipped_error(c_delta),
										 name = 'c_loss')
			self.c_learning_rate_step = tf.placeholder('int64', None,
											name='c_learning_rate_step')
			self.c_learning_rate_op = tf.maximum(
					config.c_params.learning_rate_minimum,
					tf.train.exponential_decay(
						learning_rate = config.c_params.learning_rate,
						global_step   = self.c_learning_rate_step,
						decay_steps   = config.c_params.learning_rate_decay_step,
						decay_rate    = config.c_params.learning_rate_decay,
						staircase     = True))
			self.c_optim = tf.train.RMSPropOptimizer(
								learning_rate = self.c_learning_rate_op,
								momentum      = 0.95,
								epsilon       = 0.01).minimize(self.c_loss)
	def build_hdqn(self, config):

		
		

		self.build_meta_controller(config)
		
		self.build_controller(config)

		with tf.variable_scope('summary'):
			scalar_summary_tags = ['average.reward', 'average.loss', \
						  'average.q', 'time', 'episode.max reward', \
						  'episode.min reward','episode.avg reward', \
						  'num of game', 'training.learning_rate']

			self.summary_placeholders = {}
			self.summary_ops = {}

			for tag in scalar_summary_tags:
				self.summary_placeholders[tag] = tf.placeholder('float32',
										 None, name=tag.replace(' ', '_'))
				self.summary_ops[tag]	= tf.summary.scalar("%s-/%s" % \
					(self.env_name, tag), self.summary_placeholders[tag])

			histogram_summary_tags = ['episode.rewards', 'actions', 'goals']

			for tag in histogram_summary_tags:
				self.summary_placeholders[tag] = tf.placeholder('float32',
											 None, name=tag.replace(' ', '_'))
				self.summary_ops[tag] = tf.summary.histogram(tag,
											self.summary_placeholders[tag])
			print(self.model_dir)
			self.writer = tf.summary.FileWriter('./logs/%s' % \
										   self.model_dir, self.sess.graph)
			
		tf.global_variables_initializer().run()
		
		mc_vars = list(self.mc_w.values()) + [self.mc_step_op]
		c_vars = list(self.c_w.values()) + [self.c_step_op]
		
		self._saver = tf.train.Saver(var_list = mc_vars + c_vars,
								      max_to_keep=30)

		self.load_model()
		self.update_target_q_networks()

	def update_target_q_networks(self):
		print("META")
		for name in self.mc_w.keys():
			self.mc_target_w_assign_op[name].eval(
					{self.mc_target_w_input[name]: self.mc_w[name].eval()})
		print("CONTROLLER")
		for name in self.c_w.keys():
#			print("__________")
#			print(self.c_target_w_assign_op[name])
#			print(self.c_target_w_input[name])
#			print(self.c_w[name])
			self.c_target_w_assign_op[name].eval(
					{self.c_target_w_input[name]: self.c_w[name].eval()})
	

	def save_weight_to_pkl(self):
		if not os.path.exists(self.weight_dir):
			os.makedirs(self.weight_dir)

		for name in self.w.keys():
			save_pkl(obj = self.w[name].eval(),
					path = os.path.join(self.weight_dir, "%s.pkl" % name))

	def load_weight_from_pkl(self, cpu_mode=False):
		with tf.variable_scope('load_pred_from_pkl'):
			self.w_input = {}
			self.w_assign_op = {}

			for name in self.w.keys():
				self.w_input[name] = tf.placeholder('float32',
										self.w[name].get_shape().as_list(),
										name=name)
				self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

		for name in self.w.keys():
			self.w_assign_op[name].eval(
					{self.w_input[name]: load_pkl(os.path.join(
										self.weight_dir, "%s.pkl" % name))})

		self.update_target_q_network()

	def inject_summary(self, tag_dict, step):
		summary_str_lists = self.sess.run(
					[self.summary_ops[tag] for tag in tag_dict.keys()],
					{self.summary_placeholders[tag]: value for tag, value \
														  in tag_dict.items()})
		for summary_str in summary_str_lists:
			self.writer.add_summary(summary_str, self.step)

	def play(self, n_step=10000, n_episode=100, test_ep=None, render=False):
		if test_ep == None:
			test_ep = self.ep_end

		test_history = History(self.config)

		if not self.display:
			gym_dir = '/tmp/%s-%s' % (self.env_name, get_time())
			self.env.env.monitor.start(gym_dir)

		best_reward, best_idx = 0, 0
		for idx in range(n_episode):
			screen, reward, action, terminal = self.env.new_random_game()
			current_reward = 0

			for _ in range(self.history_length):
				test_history.add(screen)

			for t in tqdm(range(n_step), ncols=70):
				# 1. predict
				action = self.predict(test_history.get(), test_ep)
				# 2. act
				screen, reward, terminal = self.env.act(action = action,
														is_training=False)
				# 3. observe
				test_history.add(screen)

				current_reward += reward
				if terminal:
					break

			if current_reward > best_reward:
				best_reward = current_reward
				best_idx = idx

			print("="*30)
			print(" [%d] Best reward : %d" % (best_idx, best_reward))
			print("="*30)

		if not self.display:
			self.env.env.monitor.close()
			#gym.upload(gym_dir, writeup='https://github.com/devsisters/DQN-tensorflow', api_key='')
