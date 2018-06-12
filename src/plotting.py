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










