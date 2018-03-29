# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from base import Epsilon
import numpy as np

class Goal(metaclass = ABCMeta):
	def __init__(self, n, name, config):
		self.n = n
		self.name = str(name)
		self.set_counter = 0
		self.achieved_counter = 0		
		
	def setup_epsilon(self, config, start_step):
		assert False, 'Not used'
		self.epsilon = Epsilon(config, start_step)
		
	@property
	def epsilon(self):
		try:
			epsilon = 1 - self.achieved_counter / self.achieved_counter
		except ZeroDivisionError:
			epsilon = 1
		return epsilon
	
	def setup_one_hot(self, length):
		one_hot = np.zeros(length)
		one_hot[self.n] = 1.
		self.one_hot = one_hot
	
	@abstractmethod
	def is_achieved(self):
		pass
	
	def finished(self, metrics, is_achieved):
		self.set_counter += 1
		self.achieved_counter += int(is_achieved)
		metrics.store_goal_result(self, is_achieved)
	
class MDPGoal(Goal):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
			
	def is_achieved(self, screen):
		return np.array_equal(screen, self.one_hot)
		