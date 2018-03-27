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
from goals import MDPGoal

class Metrics:
	def __init__(self, config):
		self.restart()
		self.mc_params = config.mc_params
		self.c_params = config.c_params
	
	def start_timer(self):
		self.t0 = time.time()
		
	def restart_timer(self):
		self.t1 = time.time()
		time_, self.t0 = self.t1 - self.t0, self.t1
		return time_
	def restart_after():
		pass
	def restart(self):
		self.num_game = 0
		self.update_count = 0
		
		
		#Scalars with c and mc versions
		scalars = ['total_reward', 'ep_reward', 'total_q',
				'total_loss']
		for scalar in scalars:
			for prefix in ['mc', 'c']:
				setattr(self, prefix + "_" + scalar, 0)
		
		self.mc_ep_rewards = []
		self.c_ep_rewards = []
		self.actions, self.goals = [], []
		
		self.start_timer()
		
	def increment_rewards(self, internal, external):
		self.c_total_reward += internal
		self.c_ep_reward += internal
		
		self.mc_total_reward += external		
		self.mc_ep_reward += external
		
	def c_inject_summaries(self, step):
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
		pass
		
	def mc_inject_summaries(self, step):
		pass
	
	def compute_test(self, prefix, update_count):
		assert prefix in ['c', 'mc']
		prefix = '_' + prefix
		config = getattr(self, prefix + 'params')
		test_step = config.test_step
		total_reward = getattr(self, prefix + 'total_reward')
		setattr(self, prefix + 'avg_reward', total_reward / test_step)
		total_loss = getattr(self, prefix + 'total_loss')
		setattr(self, prefix + 'avg_loss', total_loss / update_count)
		total_q = getattr(self, prefix + 'total_q')
		setattr(self, prefix + 'avg_q', total_q / update_count)
		
		ep_rewards = getattr(self, prefix + 'ep_rewards')
		
		try:
			setattr(self, prefix + 'max_ep_reward', np.max(ep_rewards))
			setattr(self, prefix + 'min_ep_reward', np.min(ep_rewards))
			setattr(self, prefix + 'mean_ep_reward', np.mean(ep_rewards))
		except Exception as e:
			print(str(e))
			for s in ['max', 'min', 'mean']:
				setattr(self, prefix + s +'_ep_reward', 0.)
			
	def close_episode(self):
		self.num_game += 1
		self.mc_ep_rewards.append(self.mc_ep_reward)
		self.c_ep_rewards.append(self.c_ep_reward)
		self.mc_ep_reward, self.c_ep_reward = 0., 0.
		
class Agent(BaseModel):
	def __init__(self, config, environment, sess):
		super(Agent, self).__init__(config)
		self.sess = sess
		self.weight_dir = 'weights'

		self.env = environment
		self.goals = self.define_goals(config)
		print(self.goals)
		#Update controller config with goal size
		self.config.c_params.update({'input_size': \
							self.env.goal_size + self.env.state_size})
		self.config.mc_params.update({'input_size': self.env.state_size})
		
		
		
		self.mc_history = History(self.config.mc_params)
		self.c_history = History(self.config.c_params)
		
		self.mc_memory = ReplayMemory(self.config.mc_params, self.model_dir)
		self.c_memory = ReplayMemory(self.config.c_params, self.model_dir)
		
			
	
		self.build_hdqn(config)
		
		
#	def is_goal_achieved(self, goal, state):
#		return goal.achieved(state)
	def one_hot_goal(self, goal):
		one_hot_goal = np.zeros(self.env.goal_size)
		one_hot_goal[goal.n] = 1
		return one_hot_goal
	def inverse_one_hot_goal(self, one_hot_goal):
		n = np.where(one_hot_goal)[0][0]
		goal = self.get_goal(n)
		return goal
	
	def get_goal(self, n):
		return self.goals[n]
		
	def define_goals(self, config):
		mdps = ["stochastic_mdp-v0","ez_mdp-v0","trap_mdp-v0"]
		self.env.goal_size = self.env.state_size
		
		
		
		goals = {}
		for n in range(self.env.goal_size):
			if self.env.env_name in mdps:
				goal_name = "s" + str(n)
				goal = MDPGoal(n, goal_name, config.c_params)			
				goal.setup_one_hot(self.env.goal_size)
			elif 0:
				#Space Fortress
				pass
			else:
				raise ValueError("No prior goals for " + self.env.env_name)
			goals[goal.n] = goal
		
		return goals
	
	def predict_next_goal(self, test_ep = None):
		s_t = self.mc_history.get()
		ep = test_ep or self.mc_epsilon.value
	
		if random.random() < ep:
			n_goal = random.randrange(self.env.goal_size)
		else:
			n_goal = self.mc_q_goal.eval({self.s_t: [s_t]})[0]
			
		goal = self.get_goal(n_goal)
		return goal
	

		
	def predict_next_action(self, test_ep = None):
		
		
		
		#s_t should have goals and screens concatenated
		ep = test_ep or self.current_goal.epsilon.value
	
		if random.random() < ep:
			action = random.randrange(self.env.action_size)
		else:
			screens = self.c_history.get()
			
			action = self.q_action.eval(
							{self.c_s_t: [screens],
							 self.c_g_t: [self.current_goal.one_hot]})[0]
		
		return action
	def mc_observe(self, screen, ext_reward, goal, terminal):
		params = self.config.mc_params
		self.mc_history.add(screen)
		next_state = screen
		self.mc_memory.add(next_state, ext_reward, goal, terminal)

		if self.mc_step >  params.learn_start:
			if self.mc_step % params.train_frequency == 0:
				self.mc_q_learning_mini_batch()

			if self.mc_step % params.target_q_update_step ==\
						params.target_q_update_step - 1:
				self.mc_update_target_q_network()	
	
	def c_observe(self, screen, int_reward, action, terminal):
		params = self.config.c_params
		self.c_history.add(screen)
		next_state = np.hstack([self.current_goal.one_hot, screen])

		self.c_memory.add(next_state, int_reward, action, terminal)
		
		if self.c_step > params.learn_start:
			if self.c_step % params.train_frequency == 0:
				self.c_q_learning_mini_batch()

			if self.c_step % params.target_q_update_step == params.target_q_update_step - 1:
				self.c_update_target_q_network()

	def mc_q_learning_mini_batch(self):
		if self.mc_memory.count < self.mc_history.length:
			return
		
		s_t, goal, ext_reward, s_t_plus_1, terminal = self.mc_memory.sample()
		
		q_t_plus_1 = self.mc_target_q.eval({self.mc_target_s_t: s_t_plus_1})

		terminal = np.array(terminal) + 0.
		max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
		target_q_t = (1. - terminal) * self.config.mc_params.discount * max_q_t_plus_1 + ext_reward
		
		_, q_t, loss, summary_str = self.sess.run([self.mc_optim, self.mc_q,
											 self.mc_loss, self.mc_q_summary], {
			self.mc_target_q_t: target_q_t,
			self.mc_action: goal,
			self.mc_s_t: s_t,
			self.mc_learning_rate_step: self.mc_step,
		})
		self.writer.add_summary(summary_str, self.mc_step)
		self.mc_total_loss += loss
		self.mc_total_q += q_t.mean()
		#self.mc_update_count += 1

	def c_q_learning_mini_batch(self):
		if self.c_memory.count < self.c_history.length:
			return
		
		s_t, action, int_reward, s_t_plus_1, terminal = self.c_memory.sample()
			
		#TODO: optimize goals in memory
		g_t = np.vstack([g[0] for g in s_t[:, :, :self.env.goal_size]]) 
		s_t = s_t[:, :, self.env.goal_size:]
		
		
		g_t_plus_1 = np.vstack([g[0] for g in s_t[:, :, :self.env.goal_size]])
		s_t_plus_1 = s_t_plus_1[:, :, self.env.goal_size:]
		
		
		for s,a,r,s1,t,g,g1 in zip(s_t, action, int_reward, s_t_plus_1, terminal,\
						g_t, g_t_plus_1):
			continue
			print("__________________")
			print("s_t\n",s)
			print("g_t",g)
			print("a",a)
			print("s_t1\n",s1)
			print("g_t1",g1)
			print("r",r)
			print("t",t)
			sys.exit(0)
		q_t_plus_1 = self.c_target_q.eval({
									self.c_target_s_t: s_t_plus_1,
									self.c_target_g_t: g_t_plus_1,
									 })
		
		terminal = np.array(terminal) + 0. #Boolean to float
	
		max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
		target_q_t = (1. - terminal) * self.config.c_params.discount * max_q_t_plus_1 + int_reward

		_, q_t, loss, summary_str = self.sess.run([self.c_optim, self.c_q,
											 self.c_loss, self.c_q_summary], {
			self.c_target_q_t: target_q_t,
			self.c_action: action,
			self.c_s_t: s_t,
			self.c_g_t: g_t,
			self.c_learning_rate_step: self.c_step,
		})
		self.writer.add_summary(summary_str, self.c_step)
		self.c_total_loss += loss
		self.c_total_q += q_t.mean()
		self.update_count += 1


	
	def new_episode(self):
		#screen, reward, action, terminal = self.env.new_random_game()
		screen, _, _, _ = self.env.new_game(False)
		
		self.mc_history.fill_up(screen)
		self.c_history.fill_up(screen)
		return 
	
	def train(self):
		mc_params = self.config.mc_params
		c_params = self.config.c_params
		mc_start_step = 0
		c_start_step = 0
		
		self.mc_epsilon = Epsilon(mc_params, mc_start_step)
		for key, goal in self.goals.items():
			goal.setup_epsilon(c_params, c_start_step) #TODO load individual
		
		
		self.m = Metrics(self.config)
		
		
		
		self.new_episode()
			
		
		# Initial goal
		self.current_goal = self.predict_next_goal()
		
		self.mc_step = mc_start_step
		for self.c_step in tqdm(range(c_start_step, c_params.max_step),
											  ncols=70, initial=c_start_step):
			if self.c_step == c_params.learn_start:				
				self.m.restart()
			
			# Controller acts
			action = self.predict_next_action()
#			print(self.c_history.get()[-1],', g:',self.current_goal.n,', a:', action)
				
			screen, ext_reward, terminal = self.env.act(action, is_training = True)			
			self.m.actions.append(action)
			
						
			
			# Controller learns
#			print('s',screen)
#			print('g',self.current_goal.one_hot)
			goal_achieved = self.current_goal.is_achieved(screen)
			int_reward = 1. if goal_achieved else 0.
			self.c_observe(screen, int_reward, action, terminal)

			
			self.m.increment_rewards(int_reward, ext_reward)
			
			if terminal or goal_achieved:
				# Meta-controller learns				
				self.mc_observe(screen, ext_reward, self.current_goal.n, terminal)
					
				if goal_achieved:
					pass#print("Achieved!!!")
				if terminal:
#					print(screen)
					self.m.close_episode()
					self.new_episode()
					
				self.mc_step += 1
				
				# Meta-controller sets goal
				self.current_goal = self.predict_next_goal()
				self.m.goals.append(self.current_goal.n)
				
				
			
#			print("ext",ext_reward,', int',int_reward)
			if terminal:
				pass#print("__________________________")
			
			
			if self.c_step >= c_params.c_learn_start:
				
				if self.c_step % self.c_test_step == self.c_test_step - 1:
					self.m.compute_test('c', self.c_update_count)
					self.m.compute_test('mv', self.mc_update_count)
					
					
					msg = ("\nc_avg_l: {:.4f}, mc_avg_l: {:.6f}, c_avg_q: {:.3f}, "+\
							"mc_avg_q: {:.3f}, c_avg_ep_r: {:.2f}, max_ep_r: {:.2f}, min_ep_r: "+\
							"{:.2f}, ss: {:.1f}, #g: {}").format(
									c_avg_loss, mc_avg_loss, c_avg_q,
									mc_avg_q, avg_ep_reward, max_ep_reward,
									min_ep_reward, time_, num_game)
					print(msg)
	
					if self.m.mc_max_avg_ep_reward * 0.9 <= self.m.mc_avg_ep_reward:
						self.c_step_assign_op.eval(
								{self.step_input: self.step + 1})
						self.save_model(self.step + 1)
	
						max_avg_ep_reward = max(max_avg_ep_reward,
											    avg_ep_reward)
	
					if self.c_step > 10:
						
	
					num_game = 0
					total_reward = 0.
					self.total_loss = 0.
					self.total_q = 0.
					self.update_count = 0
					ep_reward = 0.
					ep_rewards = []
					actions = []		

	def set_interfaces_lengths(self, config):	
		#TODO remove method, not useful anymore
		config.mc_params.q_input_length = self.env.state_size
		config.mc_params.q_output_length = self.env.goal_size
		config.c_params.q_input_length = self.env.state_size
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
			self.mc_q_goal= tf.argmax(self.mc_q, axis=1)
			
			q_summary = []
			avg_q = tf.reduce_mean(self.mc_q, 0)
			

			for idx in range(self.env.goal_size):
				q_summary.append(tf.summary.histogram('mc_q/%s' % idx, avg_q[idx]))
			self.mc_q_summary = tf.summary.merge(q_summary, 'mc_q_summary')

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
			#input_size = self.env.state_size + self.env.goal_size
			self.c_s_t = tf.placeholder("float",
								[None, self.c_history.length, self.env.state_size],
								name = 'c_s_t')
			shape = self.c_s_t.get_shape().as_list()
			self.c_s_t_flat = tf.reshape(self.c_s_t, [-1, reduce(
					lambda x, y: x * y, shape[1:])])
			self.c_g_t = tf.placeholder("float",
							   [None, self.env.goal_size],
							   name = 'c_g_t')
			self.c_gs_t = tf.concat([self.c_g_t, self.c_s_t_flat],
						   axis = 1,
						   name = 'c_gs_concat')
			print(self.c_g_t)
			print(self.c_s_t_flat)
			last_layer = self.c_gs_t
			print(last_layer)
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
			self.c_q_summary = tf.summary.merge(q_summary, 'c_q_summary')

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
		print("Building meta-controller")
		self.build_meta_controller(config)
		print("Building controller")
		
		self.build_controller(config)
		
		self.setup_summary()
			
		tf.global_variables_initializer().run()
		
		mc_vars = list(self.mc_w.values()) + [self.mc_step_op]
		c_vars = list(self.c_w.values()) + [self.c_step_op]
		
		
		self._saver = tf.train.Saver(var_list = mc_vars + c_vars,
								      max_to_keep=30)

		self.load_model()
		self.mc_update_target_q_network()
		self.c_update_target_q_network()
		
	def setup_summary(self):
		scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
					'test.time', 'episode.max reward', 'episode.min reward', \
					 'episode.avg reward', 'test.num of game', 'learning_rate']
		for k, goal in self.goals.items():
			name = goal.name
			rate_tag = name + '_success_rate'
			frequency_tag = 'test_' + name + '_freq'
			epsilon_tag = name + '_epsilon'
			scalar_summary_tags += [rate_tag, frequency_tag, epsilon_tag]
		
		histogram_summary_tags = ['episode.rewards', 'actions', 'goals']
		
		super().setup_summary(scalar_summary_tags, histogram_summary_tags)
		
	def mc_update_target_q_network(self):	
		for name in self.mc_w.keys():
			self.mc_target_w_assign_op[name].eval(
					{self.mc_target_w_input[name]: self.mc_w[name].eval()})
			
	def c_update_target_q_network(self):
		for name in self.c_w.keys():
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
