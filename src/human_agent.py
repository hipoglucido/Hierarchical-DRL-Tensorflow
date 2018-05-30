# -*- coding: utf-8 -*-
import gym
import sys
import numpy as np
from random import random
import cv2 # remove at one point
from time import sleep
from environment import Environment
from pynput.keyboard import KeyCode, Key, Listener
from configuration import Constants as CT
from goals import generate_SF_goals
#from constants import GAME, Games, SCRIPTS, ScriptsAIM_3_All, LIBRARY_PATH, EnableScripts,\
#    LIBRARY_NAME, FRAMESKIP, ScriptsSF_9, RECORD, RENDER_SPEED, RenderMode,\
#    DEFAULT_RENDER_MODE, ScriptsSFC_9, ScriptsSF_3, ScriptsAIM_3, ScriptsSFC_3,\
#    ScriptsAIM_9, KeyMap, ALL_COMBINATIONS, SCRIPT_LENGTH, GAME_VERSION,\
#    RENDER_MODE, RenderSpeed


import time
class HumanAgent():
    def __init__(self, config, environment):
        self.config = config
        self.config.env.update({'display_prob' : 1.})
        self.environment = environment
        
        goal_names =  \
            CT.goal_groups[self.environment.env_name][self.config.ag.goal_group]
        self.goals = generate_SF_goals(self.environment, goal_names)
        self.config.print()
        # Current key has to be initialized before first input of keyboard
        self.key_to_action = CT.key_to_action[self.config.env.env_name]
        print(self.key_to_action)
    def train(self):
        pass
    
    def graph(self):
        pass
    
    def play(self):
        on_release = self.on_release
        on_press = self.on_press
        self.current_key = 'wait'
        self.display_episode = True
        # Start listening to the keyboard
        print('  '.join(self.environment.gym.feature_names))
        with Listener(on_press=on_press, on_release = on_release): 
            while True:
                if self.display_episode:
                    self.environment.gym.render()
                    
                    
                else:
                    time.sleep(.01)
                if self.current_key == 'Key.esc':
                    self.environment.gym.close()
                elif self.current_key not in self.key_to_action:
                    self.current_key = 'wait'
                
                action = self.key_to_action[self.current_key]
                """
                ship_x_pos_x,
                ship_x_pos_y,                
                ship_y_pos_x,
                ship_y_pos_y,                
                ship_headings_x,
                ship_headings_y,
                square_x_pos_x,
                square_x_pos_y,
                square_y_pos_x,
                square_y_pos_y
                """
                info = {'goal_name'       : 'aim_at_fortress',
                        'display_episode' : self.display_episode}
                observation, reward, done, _ = self.environment.act(
                                                    action = action,
                                                    info   = info)
                msg = ''
                for feature, feature_name in zip(observation, \
                                            self.environment.gym.feature_names):
                    msg += "%s:\t%.5f\n" % (feature_name, feature)
                #observation_str = '\t  '.join([str(round(f,3)) for f in observation])
                k = self.current_key.replace('Key.', '')
                msg += '\nA:%s\nR:%.2f, T%s' \
                            % (k, reward, done)
                print(msg)
                for i, goal in self.goals.items():
                    achieved = goal.is_achieved(observation, action)
                    print("Goal %s achieved -> %s" % (goal.name, str(achieved)))
                if done == 1:
                    if self.display_episode:
                        self.environment.gym.render()
                        time.sleep(.3)
                    self.environment.new_game()
                    self.display_episode = random() < self.config.gl.display_prob
                    print('  '.join(self.environment.gym.feature_names))
                    
                    if not self.display_episode:
                        cv2.destroyAllWindows()
        
    def on_press(self, key):
        self.current_key = str(key)
        
    def on_release(self, key):
        self.current_key = 'wait'
    
        


