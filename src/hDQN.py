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
			self.c_step_input = tf.placeholder('int32', None, name='c_step_input')
			self.c_step_assign_op = self.c_step_op.assign(self.c_step_input)
			
		with tf.variable_scope('mc_step'):
			self.mc_step_op = tf.Variable(0, trainable=False, name='mc_step')
			self.mc_step_input = tf.placeholder('int32', None, name='mc_step_input')
			self.mc_step_assign_op = self.mc_step_op.assign(self.mc_step_input)
				
		
		self.build_hdqn(config)
		
		
	
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
			self.mc_q_goal= tf.argmax(self.mc_q, axis=1)
			
			q_summary = []
			avg_q = tf.reduce_mean(self.mc_q, 0)
			

			for idx in range(self.env.action_size):
				q_summary.append(tf.summary.histogram('mc_q/%s' % idx, avg_q[idx]))
			self.q_summary = tf.summary.merge(q_summary, 'mc_q_summary')

		# target network
		self.create_target(config.mc_params, prefix = 'mc')

		#MC optimizer
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
						global_step   = config.mc_params.learning_rate_step,
						decay_steps   = config.mc_params.learning_rate_decay_step,
						decay_rate    = config.mc_params.learning_rate_decay,
						staircase     = True))
			self.mc_optim = tf.train.RMSPropOptimizer(
								learning_rate = self.mc_learning_rate_op,
								momentum      = 0.95,
								epsilon       = 0.01).minimize(self.mc_loss)

	def build_controller(self, config):
		pass
	
	
	def build_hdqn(self, config):

		
		self.c_w = {}
		self.c_target_w = {}

		self.build_meta_controller(config)
#		
#		self.build_controller(config)
#
#		with tf.variable_scope('summary'):
#			scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
#					'time', 'episode.max reward', 'episode.min reward', \
#					 'episode.avg reward', 'num of game', 'training.learning_rate']
#
#			self.summary_placeholders = {}
#			self.summary_ops = {}
#
#			for tag in scalar_summary_tags:
#				self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
#				self.summary_ops[tag]	= tf.summary.scalar("%s-/%s" % (self.env_name, tag), self.summary_placeholders[tag])
#
#			histogram_summary_tags = ['episode.rewards', 'actions']
#
#			for tag in histogram_summary_tags:
#				self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
#				self.summary_ops[tag]	= tf.summary.histogram(tag, self.summary_placeholders[tag])
#			print(self.model_dir)
#			self.writer = tf.summary.FileWriter('./logs/%s' % self.model_dir, self.sess.graph)
#			
#		tf.initialize_all_variables().run()
#		self._saver = tf.train.Saver(list(self.w.values()) + [self.step_op], max_to_keep=30)
#
#		self.load_model()
#		self.update_target_q_network()

	def update_target_q_network(self):
		for name in self.w.keys():
			self.target_w_assign_op[name].eval({self.target_w_input[name]: self.w[name].eval()})

	def save_weight_to_pkl(self):
		if not os.path.exists(self.weight_dir):
			os.makedirs(self.weight_dir)

		for name in self.w.keys():
			save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))

	def load_weight_from_pkl(self, cpu_mode=False):
		with tf.variable_scope('load_pred_from_pkl'):
			self.w_input = {}
			self.w_assign_op = {}

			for name in self.w.keys():
				self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
				self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

		for name in self.w.keys():
			self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})

		self.update_target_q_network()

	def inject_summary(self, tag_dict, step):
		summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
			self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
		})
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
				screen, reward, terminal = self.env.act(action, is_training=False)
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
