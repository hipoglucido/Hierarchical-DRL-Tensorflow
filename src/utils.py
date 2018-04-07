# -*- coding: utf-8 -*-
"""
Auxiliary functions and configuration settings
"""

import os
import time
import glob
import gym
import sys
import inspect
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


def insert_dirs(dirs):
    for dir_ in dirs:
            sys.path.insert(0, dir_)
            print("Added", dir_)
            
	
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#Envs
#insert_envs_paths()

sys.path.insert(0, os.path.join(ROOT_DIR))
sys.path.insert(0, os.path.join(ROOT_DIR, 'agents'))
################################
#	 FROM DQN REPO
################################
def calc_gpu_fraction(fraction_string):
    idx, num = fraction_string.split('/')
    idx, num = float(idx), float(num)

    fraction = 1 / (num - idx + 1)
    print(" [*] GPU : %.4f" % fraction)
    return fraction

import numpy as np


if (sys.version_info[0]==2):
    import cPickle
elif (sys.version_info[0]==3):
    import _pickle as cPickle


def timeit(f):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()

        print("     [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
        return result
    return timed

def get_time():
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())

@timeit
def save_pkl(obj, path):
    with open(path, 'w') as f:
        cPickle.dump(obj, f)
        print("    [*] save %s" % path)

@timeit
def load_pkl(path):
    with open(path) as f:
        obj = cPickle.load(f)
        print("    [*] load %s" % path)
        return obj

@timeit
def save_npy(obj, path):
    np.save(path, obj)
    print("    [*] save %s" % path)

@timeit
def load_npy(path):
    obj = np.load(path)
    print("    [*] load %s" % path)
    return obj