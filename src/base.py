import os
import pprint
from configuration import MetaControllerParameters, ControllerParameters
import inspect
import tensorflow as tf
import numpy as np
import utils
import ops
from functools import reduce
from ops import linear, clipped_error

pp = pprint.PrettyPrinter().pprint



class BaseModel(object):
	"""Abstract object representing an Reader model."""
	def __init__(self, config):
		self._saver = None
		self.config = config
		for k, v in inspect.getmembers(config):
			name = k if not k.startswith('_') else k[1:]
			setattr(self, name, v)

	def add_dense_layers(self, config, input_layer, prefix):
		last_layer = input_layer
		print(last_layer)
		prefix = prefix + "_" if prefix != '' else prefix
		
		if config.activation_fn == 'relu':
			activation_fn = tf.nn.relu
		else:
			raise ValueError("Wrong activaction function")
			
		for i, neurons in enumerate(config.architecture):
			number = 'l' + str(i + 1)
			layer_name = prefix + number
			layer, weights, biases = \
				ops.linear(input_ = last_layer,
		               output_size = neurons,
					   activation_fn = activation_fn,
					   name = layer_name)
			setattr(self, layer_name, layer)
			getattr(self, prefix + 'w')[number + "_w"] = weights
			getattr(self, prefix + 'w')[number + "_b"] = biases
			last_layer = layer
#			print(layer_name, layer.get_shape().as_list(), 'added')		
			print(layer, 'added')
		return last_layer

	def create_target(self, config):
		print("Creating target...")

		prefix = config.prefix + '_' if config.prefix != '' else config.prefix
		#config = config
		#config = self.config
#		# target network
		aux1 = prefix + 'target'                         # mc_target
		aux2 = aux1 + '_s_t'                             # mc_target_s_t
		aux3 = aux1 + '_w'                               # mc_target_w
		aux4 = aux1 + '_q'                               # mc_target_q
		aux5 = 'w' if prefix == '' else prefix + 'w'    # mc_w
		target_w = {}
		
		
		setattr(self, aux3, target_w)
		with tf.variable_scope(aux1):
			target_s_t = tf.placeholder("float",
					    [None, config.history_length, config.q_input_length],
						name = aux2)
			shape = target_s_t.get_shape().as_list()
			target_s_t_flat = \
				tf.reshape(target_s_t,
					      [-1, reduce(lambda x, y: x * y, shape[1:])])
			
			last_layer = target_s_t_flat
			last_layer = self.add_dense_layers(config = config,
											   input_layer = last_layer,
											   prefix = aux1)
			
			
			
			target_q, weights, biases = \
						linear(last_layer,
							   config.q_output_length, name=aux4)
			print(target_q)
			setattr(self, aux2, target_s_t)
			getattr(self, aux3)['q_w'] = weights
			getattr(self, aux3)['q_b'] = biases
			setattr(self, aux4, target_q)
	

		with tf.variable_scope(prefix + 'pred_to_target'):
			target_w_input = {}
			target_w_assign_op = {}
			w = getattr(self, aux5)
			
			for name in w.keys():
#				print("__________________________")
				target_w_input[name] = tf.placeholder(
						       'float32',
							   target_w[name].get_shape().as_list(),
							   name=name)
				target_w_assign_op[name] = target_w[name].assign(
												value = target_w_input[name])
#				print(target_w_input[name])
#				print(target_w_assign_op[name])
		setattr(self, aux3 + "_input", target_w_input)
		setattr(self, aux3 + "_assign_op", target_w_assign_op)
		
		
		
	def save_model(self, step=None):
		print(" [*] Saving checkpoints...")

		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)
		self.saver.save(self.sess, self.checkpoint_dir, global_step=step)

	def load_model(self):
		print(" [*] Loading checkpoints...")

		ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			fname = os.path.join(self.checkpoint_dir, ckpt_name)
			self.saver.restore(self.sess, fname)
			print(" [*] Load SUCCESS: %s" % fname)
			return True
		else:
			print(" [!] Load FAILED: %s" % self.checkpoint_dir)
			return False

	@property
	def checkpoint_dir(self):
		return os.path.join('checkpoints', self.model_dir)

			
	@property
	def model_dir(self):
		parts = self.config.as_list(ignore = True)
		
		result = os.path.join(*parts)
		
		return result

	@property
	def saver(self):
		if self._saver == None:
			self._saver = tf.train.Saver(max_to_keep=10)
		return self._saver




