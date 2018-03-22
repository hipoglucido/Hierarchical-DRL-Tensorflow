import numpy as np

class History:
	def __init__(self, config):
		
		self.history = np.zeros(
				[config.history_length, config.state_size], dtype=np.float32)
	@property
	def length(self):
		return self.get().shape[0]
	
	def add(self, screen):
		self.history[:-1] = self.history[1:]
		self.history[-1] = screen

	def reset(self):
		self.history *= 0

	def get(self):
		return self.history
	
	def fill_up(self, state):
		for _ in range(self.length):
			self.add(state)