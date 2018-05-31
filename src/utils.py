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
import argparse
import tensorflow as tf
from configuration import Constants as CT
import math
from tensorflow.contrib.layers.python.layers import initializers
from PIL import Image

################################
#	 AUXILIARY FUNCTIONS
################################
def pp(prefix, word):
    """
    Pp stands for preprend prefix
    """
    return prefix + word
    
def clamp(n, smallest, largest): return max(smallest, min(n, largest))

def get_timestamp():
	#return time.strftime("%Ss%Hh%Mm%b%d")
    return time.strftime("%dd%Hh%Mm%Ss")
#	return time.strftime("%Hh%Mm%Ss")


def insert_dirs(dirs):
    for dir_ in dirs:
            sys.path.insert(0, dir_)
            #print("Added", dir_)
            
	
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def revert_cyclic_feature(X_sin, X_cos, is_scaled, scale_after):
    if is_scaled:
        # [0, 1] -> [-1, 1]
        X_sin, X_cos = X_sin * 2 - 1, X_cos * 2 - 1
    if X_sin > 0:
        X = math.acos(X_cos)
    else:
        X = CT.c - math.acos(X_cos)
        
    if scale_after:
        # [0, 2pi] -> [0, 1]
        X /= CT.c
    return X
    

################################
#	 FROM DQN REPO
################################
def calc_gpu_fraction(fraction_string):
    idx, num = fraction_string.split('/')
    idx, num = float(idx), float(num)

    fraction = 1 / (num - idx + 1)
#    print(" [*] GPU : %.4f" % fraction)
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



################################
#	 OPS
################################


def linear(input_, output_size, stddev=0.0002, bias_start=0.0,
    activation_fn = None, name = 'linear'):
    
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
                            tf.contrib.layers.xavier_initializer(uniform = True))
        b = tf.get_variable('bias', [output_size], 
                initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(input_, w), b)
#        out = tf.matmul(input_, w)
    if activation_fn != None:
        return activation_fn(out), w, b
    else:
        return out, w, b

"""Loss functions."""




def huber_loss(TD, weights = None, max_grad=1.):
    """Calculate the huber loss.
    See https://en.wikipedia.org/wiki/Huber_loss
    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.
    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    #a = tf.abs(y_true - y_pred)
    less_than_max = 0.5 * tf.square(TD)
    greater_than_max = max_grad * (TD - 0.5 * max_grad)
    return tf.where(TD <= max_grad, x=less_than_max, y=greater_than_max)



def mean_huber_loss(TD, max_grad=1.):
    """Return mean huber loss.
    Same as huber_loss, but takes the mean over all values in the
    output tensor.
    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.
    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    return tf.reduce_mean(huber_loss(TD, max_grad=max_grad))


def weighted_huber_loss(TD, weights, max_grad=1.):
    """Return mean huber loss.
    Same as huber_loss, but takes the mean over all values in the
    output tensor.
    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    weights: np.array, tf.Tensor
      weights value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.
    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    return tf.reduce_mean(weights*huber_loss(TD, max_grad=max_grad))

    

    








