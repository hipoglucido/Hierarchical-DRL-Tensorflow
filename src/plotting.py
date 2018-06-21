# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 11:49:49 2018

@author: victorgarcia
"""


import configuration
import os
import tensorflow as tf
import glob
import pandas as pd
from matplotlib import pyplot as plt
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    y = np.array(y)
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def get_single_series(path_to_events_file, scalar_key):
    model_name = path_to_events_file.split(os.path.sep)[-2]
    print("\tGetting %s from %s" % (scalar_key, model_name))
    steps = []
    values = []
    for e in tf.train.summary_iterator(path_to_events_file):   
        for v in e.summary.value:
            if scalar_key in v.tag:
                values.append(v.simple_value)
                steps.append(e.step)
    series = pd.Series(index = steps, data = values)
    return series

def get_all_series_averaged(folder_key, scalar_key, experiment_folder,
                                                                steps_mode):
    print("Searching for folder_key='%s'" % folder_key)
    series_list = []
    run_paths = [os.path.join(experiment_folder, f) for f in \
                             os.listdir(experiment_folder) if folder_key in f]
    training_times = []
    for run_path in run_paths:
        path_to_events_file = glob.glob(os.path.join(run_path,'events.out*'))[0]
        series = get_single_series(path_to_events_file, scalar_key)
        try:
            with open(os.path.join(run_path,'total_training_seconds.txt')) as fp:
                content = fp.read()
            training_time = float(content) / 3600
            training_times.append(training_time)
        except:
            print("Not finished")
        series_list.append(series)
    df = pd.DataFrame(series_list)
    means = df.mean(axis = 0)
    stds = df.std(axis = 0)
    steps = df.columns.to_series()
    
    if steps_mode == 'millions':
        steps /= 1e6
    elif steps_mode == 'thousands':
        steps /= 1e3
    try:
        avg_training_time = sum(training_times) / len(training_times)
    except:
        avg_training_time = -1
    data = {'means' : means,
            'stds'  : stds,
            'steps' : steps,
            'time'  : avg_training_time}
                
    return data
gl = configuration.GlobalSettings()
#
#experiment_folder = os.path.join(gl.logs_dir, 'exp1')
def plot1():
    """
    TOY PROBLEM 1
    """
    experiment_folder = os.path.join(gl.logs_dir, 'exp1')
    steps_mode = 'thousands'
    data_hdqn = get_all_series_averaged(folder_key = 'athdqn',
                                       scalar_key = 'mc_avg_ep_reward',
                                       experiment_folder = experiment_folder,
                                       steps_mode = steps_mode)
    data_dqn = get_all_series_averaged(folder_key = 'atdqn',
                                       scalar_key = 'avg_ep_reward',
                                       experiment_folder = experiment_folder,
                                       steps_mode = steps_mode)
    datas = [data_dqn, data_hdqn]
    labels = ['DQN', 'hDQN']
    colors = ['red', 'blue']
    alpha = .5
    linewidth = 0
    fontsize = 15
    plt.figure()
    for data, label, color in zip(datas, labels, colors):
        err, means, steps = data['stds'], data['means'], data['steps']
        plt.plot(steps, means, label = label, color = color)
        plt.fill_between(steps,
                         means - err,
                         means + err,
                         alpha = alpha, color = color,linewidth = linewidth)
    plt.ylabel('Average episode $R$', fontsize = fontsize)
    plt.xlabel('%s of steps' % steps_mode.capitalize(), fontsize = fontsize)
    plt.legend(fontsize = fontsize,
               loc = 'lower right')
    plt.show()


def get_datas(folder_keys, scalar_key, experiment_folder, steps_mode):
    datas = []
    for folder_key in folder_keys:
        data = get_all_series_averaged(folder_key = folder_key,
                                       scalar_key = scalar_key,
                                       experiment_folder = experiment_folder,
                                       steps_mode = steps_mode)
        datas.append(data)    
    return datas

def extensions():
    """
    DQN EXTENSIONS
    """
    experiment_folder = '/home/victorgarcia/Downloads/logs/20d15h40m11s_extensions_exp'
    steps_mode = 'millions'
    scalar_key = 'avg_ep_reward'
    labels = ['Vanilla', 'Double Q', 'Dueling', 'Prioritized Replay', 'Rainbow']
    folder_keys = [ 'dq0_d0_p0',  'dq1_d0_p0', 'dq0_d1_p0', 'dq0_d0_p1', 'dq1_d1_p1']
    datas = get_datas(folder_keys, scalar_key, experiment_folder, steps_mode)
    
    alpha = .3
    linewidth = 0
    fontsize = 15
    plt.figure(figsize = (12, 7))
    for data, label in zip(datas, labels):
        err, means, steps, t = data['stds'], data['means'], data['steps'], data['time']
        if len(steps) == 0:
                continue
        means = savitzky_golay(means, 51, 3)
        final_label = "(%.1fh) %s" % (t, label)
        plt.plot(steps, means, label = final_label, linewidth = 2)
        plt.fill_between(steps,
                         means - err,
                         means + err,
                         alpha = alpha,linewidth = linewidth)
    plt.ylabel('Average episode $R$', fontsize = fontsize)
    plt.xlabel('%s of steps' % steps_mode.capitalize(), fontsize = fontsize)
    plt.legend(fontsize = fontsize,
               loc = 'lower right')
    plt.show()
def architectures():
    """
    ARCHITECTURES
    """
    
    experiment_folder = '/home/victorgarcia/Downloads/logs/20d15h40m05s_architectures_exp'
    steps_mode = 'millions'
    scalar_key = 'avg_ep_reward'
    architectures = [[16],
                    [64, 64],
                    [64, 64, 64, 64],
                    [512],
                    [512, 512],
                    [512, 512, 512, 512]]
    labels = ['-'.join([str(n) for n in a]) for a in architectures]
    folder_keys = ['a' + l + '_' for l in labels]
    colors = ['black', 'red', 'blue', 'green', 'orange']
    datas = get_datas(folder_keys, scalar_key, experiment_folder, steps_mode)
    
    alpha = .6
    linewidth = 0
    fontsize = 15
    plt.figure(figsize = (12,7))
    for data, label, color in zip(datas, labels, colors):
        err, means, steps, t = data['stds'], data['means'], data['steps'], data['time']
        if len(steps) == 0:
            continue
        means = savitzky_golay(means, 51, 3)
        final_label = "(%.1fh) %s" % (t, label)
        plt.plot(steps, means, label = final_label, color = color, linewidth = 1,
                 alpha = alpha)
        plt.fill_between(steps,
                         means - err,
                         means + err,
                         alpha = .5 * alpha, color = color,linewidth = linewidth)
    plt.ylabel('Average episode $R$', fontsize = fontsize)
    plt.xlabel('%s of steps' % steps_mode.capitalize(), fontsize = fontsize)
    plt.legend(fontsize = .7 * fontsize,
               loc = 'lower right')
    plt.show()

def actionrepeats():
    """
    ACTION REPEATS
    """
    
    experiment_folder = '/home/victorgarcia/Downloads/logs/20d15h40m02s_action_repeat_exp'
    steps_mode = 'millions'
    scalar_key = 'avg_ep_reward'
    actions_repeats = list(range(1, 11))
    labels = ["ar=%d" % ar for ar in actions_repeats]
    folder_keys = ["ar%d_" % ar for ar in actions_repeats]
    #colors = ['black', 'red', 'blue', 'green', 'orange']
    datas = get_datas(folder_keys, scalar_key, experiment_folder, steps_mode)
    
    alpha = 1
    linewidth = 0
    fontsize = 15
    plt.figure(figsize = (12,7))
    for data, label in zip(datas, labels):
        err, means, steps = data['stds'], data['means'], data['steps']
        if len(steps) == 0:
            continue
        means = savitzky_golay(means, 51, 3)
        plt.plot(steps, means, label = label, linewidth = 1,
                 alpha = alpha)
        plt.fill_between(steps,
                         means - err,
                         means + err,
                         alpha = .1 * alpha,linewidth = linewidth)
    plt.ylabel('Average episode $R$', fontsize = fontsize)
    plt.xlabel('%s of steps' % steps_mode.capitalize(), fontsize = fontsize)
    plt.legend(fontsize = .7 * fontsize,
               loc = 'lower right')
    plt.ylim([-20,20])
    plt.show()
    




