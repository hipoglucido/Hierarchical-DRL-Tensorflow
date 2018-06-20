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

def get_all_series_averaged(folder_key, scalar_key, experiment_folder, steps_mode):
    print("Searching for folder_key='%s'" % folder_key)
    series_list = []
    run_paths = [os.path.join(experiment_folder, f) for f in os.listdir(experiment_folder) if folder_key in f]
    for run_path in run_paths:
        path_to_events_file = glob.glob(os.path.join(run_path,'events.out*'))[0]
        series = get_single_series(path_to_events_file, scalar_key)
        series_list.append(series)
    df = pd.DataFrame(series_list)
    means = df.mean(axis = 0)
    stds = df.std(axis = 0)
    steps = df.columns.to_series()
    
    if steps_mode == 'millions':
        steps /= 1e6
    elif steps_mode == 'thousands':
        steps /= 1e3
    
    data = {'means' : means,
            'stds'  : stds,
            'steps' : steps}
                
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


"""
TOY PROBLEM 1
"""
experiment_folder = '/home/victorgarcia/Downloads/logs'
steps_mode = 'millions'
scalar_key = 'avg_ep_reward'
labels = ['Vanilla', 'Double Q', 'Dueling', 'Prioritized Replay']
folder_keys = [ 'dq0_d0_p0',  'dq1_d0_p0', 'dq0_d1_p0', 'dq0_d0_p1']
colors = ['black', 'red', 'blue', 'green']
datas = []
for folder_key in folder_keys:
    data = get_all_series_averaged(folder_key = folder_key,
                                   scalar_key = scalar_key,
                                   experiment_folder = experiment_folder,
                                   steps_mode = steps_mode)
    datas.append(data)

alpha = .5
linewidth = 0
fontsize = 15
plt.figure()
for data, label, color in zip(datas, labels, colors):
    err, means, steps = data['stds'], data['means'], data['steps']
    plt.plot(steps, means, label = label, color = color, linewidth = 1)
    plt.fill_between(steps,
                     means - err,
                     means + err,
                     alpha = alpha, color = color,linewidth = linewidth)
plt.ylabel('Average episode $R$', fontsize = fontsize)
plt.xlabel('%s of steps' % steps_mode.capitalize(), fontsize = fontsize)
plt.legend(fontsize = fontsize,
           loc = 'lower right')
plt.show()










