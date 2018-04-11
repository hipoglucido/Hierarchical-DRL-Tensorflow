import random
import logging
import numpy as np
logger = logging.getLogger(__name__)
from state_buffer import StateBuffer

class Agent:
	def __init__(self, environment, replay_memory, deep_q_network, args):
		self.env = environment
		self.mem = replay_memory
		self.net = deep_q_network
		self.buf = StateBuffer(args)
		self.num_actions = self.env.numActions()
		print(self.num_actions)
		self.random_starts = args.random_starts
		self.history_length = args.history_length

		self.exploration_rate_start = args.exploration_rate_start
		self.exploration_rate_end = args.exploration_rate_end
		self.exploration_decay_steps = args.exploration_decay_steps
		self.exploration_rate_test = args.exploration_rate_test
		self.total_train_steps = args.start_epoch * args.train_steps

		self.train_frequency = args.train_frequency
		self.train_repeat = args.train_repeat

		self.callback = None

	def _restartRandom(self):
		self.env.restart()
		tries = 3
		# perform random number of dummy actions to produce more stochastic games
		while tries:
			try:
				for i in xrange(random.randint(self.history_length, self.random_starts) + 1):
					reward = self.env.act(0)
					screen = self.env.getScreen()
					terminal = self.env.isTerminal()
					# assert not terminal, "terminal state occurred during random initialization"
					# add dummy states to buffer
					tries = 0
					self.buf.add(screen)
			except Exception, e:
				print(e)
				tries -= 1
				if tries <= -1:
					assert not terminal, "terminal state occurred during random initialization"
#					pass


	def _explorationRate(self):
		# calculate decaying exploration rate
		if self.total_train_steps < self.exploration_decay_steps:
			return self.exploration_rate_start - self.total_train_steps * (self.exploration_rate_start - self.exploration_rate_end) / self.exploration_decay_steps
		else:
			return self.exploration_rate_end

	def step(self, exploration_rate):
		# exploration rate determines the probability of random moves
		if random.random() < exploration_rate:
			action = random.randrange(self.num_actions)
#			print("Random action = %d" % action)
			logger.debug("Random action = %d" % action)
		else:
			# otherwise choose action with highest Q-value
			state = self.buf.getStateMinibatch()
			# for convenience getStateMinibatch() returns minibatch
			# where first item is the current state
			qvalues = self.net.predict(state)
			assert len(qvalues[0]) == self.num_actions
			# choose highest Q-value of first state
			action = np.argmax(qvalues[0])
			# print(qvalues[0])
			logger.debug("Predicted action = %d" % action)

		# perform the action
		reward = self.env.act(action)
		screen = self.env.getScreen()
		terminal = self.env.isTerminal()

		# print reward
		if reward <> 0:
			logger.debug("Reward: %d" % reward)

		# add screen to buffer
		self.buf.add(screen)

		# restart the game if over
		if terminal:
			logger.debug("Terminal state, restarting")
			self._restartRandom()

		# call callback to record statistics
		if self.callback:
			self.callback.on_step(action, reward, terminal, screen, exploration_rate)

		return action, reward, screen, terminal

	def play_random(self, random_steps):
		# play given number of steps
		for i in xrange(random_steps):
			# use exploration rate 1 = completely random
			self.step(1)

	def train(self, train_steps, epoch = 0):
		# do not do restart here, continue from testing
		#self._restartRandom()
		# play given number of steps
		for i in xrange(train_steps):
			# perform game step
			action, reward, screen, terminal = self.step(self._explorationRate())
			self.mem.add(action, reward, screen, terminal)
			# train after every train_frequency steps
			if self.mem.count > self.mem.batch_size and i % self.train_frequency == 0:
				# train for train_repeat times
				for j in xrange(self.train_repeat):
					# sample minibatch
					minibatch = self.mem.getMinibatch()
#					a1 = minibatch[0][0][0]
#					a2 = minibatch[3][0][0]
#					a3 = minibatch[0][0][1]
#					a4 = minibatch[3][0][1]
#					print(minibatch[0].shape)
#					print(minibatch[3].shape)
#					import cv2
#					cv2.imshow("1",a1)
##					print(zip(minibatch[1], minibatch[2], minibatch[4]))
#					cv2.waitKey(0)
#					cv2.imshow("1",a2)
#					cv2.waitKey(0)
#					cv2.imshow("1",a3)
#					cv2.waitKey(0)
#					cv2.imshow("1",a4)
#					cv2.waitKey(0)
#					import sys
#					sys.exit(0)
					# train the network
					self.net.train(minibatch, epoch)
			# increase number of training steps for epsilon decay
			self.total_train_steps += 1

	def test(self, test_steps, epoch = 0):
		# just make sure there is history_length screens to form a state
		self._restartRandom()
		# play given number of steps
		for i in xrange(test_steps):
			# perform game step
			self.step(self.exploration_rate_test)

	def play(self, num_games):
		# just make sure there is history_length screens to form a state
		self._restartRandom()
		for i in xrange(num_games):
			# play until terminal state
			terminal = False
			while not terminal:
				action, reward, screen, terminal = self.step(self.exploration_rate_test)
				# add experiences to replay memory for visualization
				self.mem.add(action, reward, screen, terminal)

class PerfectAgent(Agent):

	def __init__(self, environment, replay_memory, deep_q_network, args):
		Agent.__init__(self, environment, replay_memory, deep_q_network, args)
		self.randomGameInterval = 12
		self.performMax = self.randomGameInterval
		self.randomPlaysMax = 70
		self.randomPlays = self.randomPlaysMax # How many random steps to take every random interval
		self.escapeRate = 0.06;

	def _restartRandom(self):
		self.env.restart()

	def play_random(self, random_steps):
		# play given number of steps
		for i in xrange(random_steps):
			# use exploration rate 1 = completely random
			Agent.step(self,1)

	def step(self, exploration_rate):
		# exploration rate determines the probability of random moves
		if random.random() < exploration_rate:
			# Secretly, the agent will not perform a random move,
			# but approx. the best move possible
			if self.performMax > 0:
				# Don't let the agent get stuck
				if random.random() > self.escapeRate:
					action = self.env.gym.best_action()
				else:
					action = random.randrange(self.num_actions)
			elif self.performMax == 0:
				action = random.randrange(self.num_actions)
			else: # vary between perfect and random
				if self.randomPlays:
					self.randomPlays -= 1
					action = random.randrange(self.num_actions)
				else:
					action = self.env.gym.best_action()
		else:
			# otherwise choose action with highest Q-value
			state = self.buf.getStateMinibatch()
			# for convenience getStateMinibatch() returns minibatch
			# where first item is the current state
			qvalues = self.net.predict(state)
			assert len(qvalues[0]) == self.num_actions
			# choose highest Q-value of first state
			action = np.argmax(qvalues[0])
			logger.debug("Predicted action = %d" % action)

		# perform the action
		reward = self.env.act(action)
		screen = self.env.getScreen()
		terminal = self.env.isTerminal()


		# print reward
		if reward <> 0:
			logger.debug("Reward: %d" % reward)

		# add screen to buffer
		self.buf.add(screen)

		# restart the game if over
		if terminal:
			self.performMax -= 1
			if self.performMax <= -2:
				self.randomPlays = self.randomPlaysMax
				self.performMax = self.randomGameInterval
			logger.debug("Terminal state, restarting")
			self._restartRandom()

		# call callback to record statistics
		if self.callback:
			self.callback.on_step(action, reward, terminal, screen, exploration_rate)

		return action, reward, screen, terminal
