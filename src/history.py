import numpy as np
#from configuration import DQNConfiguration, ControllerParameters, MetaControllerParameters

class History:
	def __init__(self, length_, size):
		
		
		self.history = np.zeros(
				[length_, size], dtype=np.float32)
	@property
	def length(self):
		return self.get().shape[0]
	

	def add(self, item):
		self.history[:-1] = self.history[1:]
		self.history[-1] = item

	def reset(self):
		self.history *= 0

	def get(self):
		return self.history
	
	def fill_up(self, item):
		for _ in range(self.length):
			self.add(item)