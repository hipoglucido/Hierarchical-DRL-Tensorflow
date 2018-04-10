# -*- coding: utf-8 -*-
import gym
import sys
import numpy as np
from random import random
import cv2 # remove at one point
from time import sleep
from environment import Environment
from pynput.keyboard import KeyCode, Key, Listener
from configuration import RenderSpeed
#from constants import GAME, Games, SCRIPTS, ScriptsAIM_3_All, LIBRARY_PATH, EnableScripts,\
#    LIBRARY_NAME, FRAMESKIP, ScriptsSF_9, RECORD, RENDER_SPEED, RenderMode,\
#    DEFAULT_RENDER_MODE, ScriptsSFC_9, ScriptsSF_3, ScriptsAIM_3, ScriptsSFC_3,\
#    ScriptsAIM_9, KeyMap, ALL_COMBINATIONS, SCRIPT_LENGTH, GAME_VERSION,\
#    RENDER_MODE, RenderSpeed



class HumanAgent():
    def __init__(self, config, environment):
        self.config = config
        self.environment = environment
        self.config.print()
        # Current key has to be initialized before first input of keyboard
        self.current_key = 0
        self.set_key_to_action()
        
    def train(self):
        pass
    def graph(self):
        pass
    def play(self):
        on_release = lambda x: None
        on_press = self.on_press
        # Start listening to the keyboard
        with Listener(on_press=on_press, on_release = on_release) as listener:
            # Stop listening if sample actions should be used instead of keyboard
            if self.config.env.render_speed != RenderSpeed.DEBUG:
                print(999)
                listener.stop()
             
            while True:
                self.environment.gym.render()
                if self.config.env.render_speed == RenderSpeed.DEBUG:
                    action = self.current_key
                    
                else:
                    action = self.environment.gym.action_space.sample()
                observation, reward, done = self.environment.act(action)
                
             
                
                if done == 1:
                    self.environment.new_game()
    def set_key_to_action(self):
        if self.environment.env_name == "SFC-v0":
            self.key_to_action = {"Key.left" : 0, "Key.up" : 2, "Key.right" : 1} 
            
        elif self.environment.env_name == "SF-v0" or self.environment.env_name == "SFS":
            self.key_to_action = {"Key.left" : 0, "Key.up" : 1, "Key.right" : 2,
                             "Key.down" : 3, "Key.space" : 3} 
            
        elif self.environment.env_name == "AIM-v0":
            self.key_to_action = {"Key.left" : 1, "Key.up" : 0, "Key.right" : 2,
                             "Key.down" : 0, "Key.space" : 0}
        else:
            assert False                   
    def on_press(self, key):
        key = str(key).replace("'", "")  # Get keyboard input
    
          
        if key in self.key_to_action.keys():
            self.current_key = self.key_to_action[key]
    
        # Do nothing on release 
    
        


