# -*- coding: utf-8 -*-
"""
Auxiliary functions and configuration settings
"""

import os
import time
import glob
import gym
import sys

################################
#	 CONFIGURATION SETTINGS
################################

#Running settigns

#Monitoring settings
USE_TB = 0

#Dir settings
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ENVS_DIR = os.path.join(ROOT_DIR, '..', 'Environments')


################################
#	 AUXILIARY FUNCTIONS
################################

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

def get_timestamp():
	return time.strftime("%Ss%Hh%Mm%b%d")

def insert_envs_paths():
	for env_dir in glob.glob(os.path.join(ENVS_DIR, '*')):
			print("Added", env_dir)
			sys.path.insert(0, env_dir)

class Logger:
	# Using the existing Logger library was throwing some issues when developing on Spyder
	def __init__(self):
		return
	
	def print(self, text):
		lines = text.split('\n')
		ts = get_timestamp()
		output = '\n'.join([ts + " -" + self.mode +"- " + l for l in lines])
		print(output)
		 
	def warn(self, text):
		self.mode = 'WARNING'
		self.print(text)
		 
	def info(self, text):
		self.mode = 'info'
		self.print(text)

def info(text): logger.info(text)
def warn(text): logger.warn(text)

def get_env(env_name):
	import gym_stochastic_mdp			
	env = gym.make(env_name)
	
	return env.env
	

#Auxiliary
logger = Logger()
#Session timestamp
SESSION_TS = get_timestamp()


if USE_TB:
	TB_DIR = os.path.join(DATA_DIR, 'tensorboard', SESSION_TS)
	os.makedirs(TB_DIR)
	logger.info("TB folder created " + TB_DIR)

#Envs
insert_envs_paths()

sys.path.insert(0, os.path.join(ROOT_DIR))
sys.path.insert(0, os.path.join(ROOT_DIR, 'agents'))
################################
#	 FROM DQN REPO
################################


import numpy as np


if (sys.version_info[0]==2):
  import cPickle
elif (sys.version_info[0]==3):
  import _pickle as cPickle

try:
  from scipy.misc import imresize
except:
  import cv2
  imresize = cv2.resize

def rgb2gray(image):
  return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def timeit(f):
  def timed(*args, **kwargs):
    start_time = time.time()
    result = f(*args, **kwargs)
    end_time = time.time()

    print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
    return result
  return timed

def get_time():
  return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())

@timeit
def save_pkl(obj, path):
  with open(path, 'w') as f:
    cPickle.dump(obj, f)
    print("  [*] save %s" % path)

@timeit
def load_pkl(path):
  with open(path) as f:
    obj = cPickle.load(f)
    print("  [*] load %s" % path)
    return obj

@timeit
def save_npy(obj, path):
  np.save(path, obj)
  print("  [*] save %s" % path)

@timeit
def load_npy(path):
  obj = np.load(path)
  print("  [*] load %s" % path)
  return obj