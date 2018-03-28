from __future__ import print_function
import os
import time
import random
import numpy as np
from tqdm import tqdm
from functools import reduce
import tensorflow as tf
import sys


from base import BaseModel, Epsilon
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
		config.update({'input_size' : self.env.state_size})
		self.history = History(self.config)
		self.memory = ReplayMemory(self.config, self.model_dir)
		
		with tf.variable_scope('step'):
			self.step_op = tf.Variable(0, trainable=False, name='step')
			self.step_input = tf.placeholder('int32', None, name='step_input')
			self.step_assign_op = self.step_op.assign(self.step_input)
			
		config.q_input_length = self.env.state_size
		config.q_output_length = self.env.action_size		
		
		self.build_dqn(config)

	
	def train(self):
		start_step = self.step_op.eval() #TODO understand, why this?		
		self.epsilon = Epsilon(self.config, start_step)
		
		num_game, self.update_count, ep_reward = 0, 0, 0.
		total_reward, self.total_loss, self.total_q = 0., 0., 0.
		max_avg_ep_reward = 0
		ep_rewards, actions = [], []

		#screen, reward, action, terminal = self.env.new_random_game()
		screen, _, _, _ = self.env.new_game(False)	
			
		self.history.fill_up(screen)
		t_0 = time.time()
		for self.step in tqdm(range(start_step, self.max_step), ncols=70,
						initial=start_step):
#		for self.step in range(start_step, self.max_step):
			
			self.epsilon.plus_one()
			if self.step == self.learn_start:
				
				num_game, self.update_count, ep_reward = 0, 0, 0.
				total_reward, self.total_loss, self.total_q = 0., 0., 0.
				ep_rewards, actions = [], []
			
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
				
				self.history.fill_up(screen)
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
								'test.time': time_,
								'episode.max reward': max_ep_reward,
								'episode.min reward': min_ep_reward,
								'episode.avg reward': avg_ep_reward,
								'test.num of game': num_game,
								'episode.rewards': ep_rewards,
								'actions': actions,
								'learning_rate': \
									self.learning_rate_op.eval(
										{self.learning_rate_step: self.step}),
								'epsilon': self.epsilon.value
							}, self.step)
	
					num_game = 0
					total_reward = 0.
					self.total_loss = 0.
					self.total_q = 0.
					self.update_count = 0
					ep_reward = 0.
					ep_rewards = []
					actions = []

	def predict(self, s_t, test_ep = None):

		ep = test_ep or self.epsilon.value
		if random.random() < ep:
			action = random.randrange(self.env.action_size)
		else:
			action = self.q_action.eval({self.s_t: [s_t]})[0]

		return action

	def observe(self, screen, reward, action, terminal):
		#reward = max(self.min_reward, min(self.max_reward, reward)) #TODO understand

		self.history.add(screen)
		self.memory.add(screen, reward, action, terminal)

		if self.step > self.learn_start:
			if self.step % self.train_frequency == 0:
				self.q_learning_mini_batch()

			if self.step % self.target_q_update_step == \
											self.target_q_update_step - 1:
				self.update_target_q_network()

	def q_learning_mini_batch(self):
		if self.memory.count < self.history_length:
			return
		
		s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()
		
		q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})

		terminal = np.array(terminal) + 0.
		max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
		target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

		_, q_t, loss, summary_str = self.sess.run([self.optim, self.q,
											 self.loss, self.q_summary], {
			self.target_q_t: target_q_t,
			self.action: action,
			self.s_t: s_t,
			self.learning_rate_step: self.step,
		})
		self.writer.add_summary(summary_str, self.step)
		self.total_loss += loss
		self.total_q += q_t.mean()
		self.update_count += 1

	def build_dqn(self, config):
		self.w = {}
		

		# training network
		with tf.variable_scope('prediction'):
			
			# tf Graph input
			self.s_t = tf.placeholder("float",
								    [None, config.history_length,
									  config.q_input_length], name='s_t')
			print(self.s_t)
			shape = self.s_t.get_shape().as_list()
			self.s_t_flat = tf.reshape(self.s_t, [-1, reduce(
											lambda x, y: x * y, shape[1:])])
			
			last_layer = self.s_t_flat
			last_layer = self.add_dense_layers(config = config,
											   input_layer = last_layer,
											   prefix = config.prefix)
			self.q, self.w['q_w'], self.w['q_b'] = linear(last_layer,
												  self.env.action_size,
												  name='q')
			self.q_action = tf.argmax(self.q, axis=1)
			
			q_summary = []
			avg_q = tf.reduce_mean(self.q, 0)
	
			
			for idx in range(self.env.action_size):
				q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
			self.q_summary = tf.summary.merge(q_summary, 'q_summary')

		self.create_target(config = config)


	
		# optimizer
		with tf.variable_scope('optimizer'):
			self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
			self.action = tf.placeholder('int64', [None], name='action')

			action_one_hot = tf.one_hot(self.action, self.env.action_size,
									   1.0, 0.0, name = 'action_one_hot')
			q_acted = tf.reduce_sum(self.q * action_one_hot,
								   reduction_indices = 1, name = 'q_acted')

			delta = self.target_q_t - q_acted



			self.loss = tf.reduce_mean(clipped_error(delta), #*
													  name='loss')
			self.learning_rate_step = tf.placeholder('int64', None, #*
											name='learning_rate_step')
			self.learning_rate_op = tf.maximum(#*
					config.learning_rate_minimum,
					tf.train.exponential_decay(
							config.learning_rate,
							self.learning_rate_step,
							config.learning_rate_decay_step,
							config.learning_rate_decay,
							staircase=True))
			self.optim = tf.train.RMSPropOptimizer(
								self.learning_rate_op, momentum=0.95,
								epsilon=0.01).minimize(self.loss)
		
		scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
					'test.time', 'episode.max reward', 'episode.min reward', \
					 'episode.avg reward', 'test.num of game', 'learning_rate', \
					 'epsilon']
		histogram_summary_tags = ['episode.rewards', 'actions']
		
		self.setup_summary(scalar_summary_tags, histogram_summary_tags)
		tf.global_variables_initializer().run()
		self._saver = tf.train.Saver(list(self.w.values()) + [self.step_op],
								   max_to_keep=30)

		self.load_model()
		self.update_target_q_network()

	def update_target_q_network(self):
		for name in self.w.keys():
			self.target_w_assign_op[name].eval(
							{self.target_w_input[name]:self.w[name].eval()})

	def save_weight_to_pkl(self):
		if not os.path.exists(self.weight_dir):
			os.makedirs(self.weight_dir)

		for name in self.w.keys():
			save_pkl(self.w[name].eval(),
							os.path.join(self.weight_dir, "%s.pkl" % name))

	def load_weight_from_pkl(self, cpu_mode=False):
		with tf.variable_scope('load_pred_from_pkl'):
			self.w_input = {}
			self.w_assign_op = {}

			for name in self.w.keys():
				self.w_input[name] = tf.placeholder('float32',
							self.w[name].get_shape().as_list(), name=name)
				self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

		for name in self.w.keys():
			self.w_assign_op[name].eval({self.w_input[name]: load_pkl(
							os.path.join(self.weight_dir, "%s.pkl" % name))})

		self.update_target_q_network()


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
				screen, reward, terminal = self.env.act(action,
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
