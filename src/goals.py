# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from base import Epsilon
import numpy as np

class Goal(metaclass = ABCMeta):
	def __init__(self, n, name, config):
		self.n = n
		self.name = str(name)			
		
	def setup_epsilon(self, config, start_step):
		self.epsilon = Epsilon(config, start_step)
	
	def setup_one_hot(self, length):
		one_hot = np.zeros(length)
		one_hot[self.n] = 1.
		self.one_hot = one_hot
	
	@abstractmethod
	def is_achieved(self):
		pass
	
class MDPGoal(Goal):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
			
	def is_achieved(self, screen):
		return np.array_equal(screen, self.one_hot)
		