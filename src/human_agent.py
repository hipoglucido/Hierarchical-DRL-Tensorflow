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
#from constants import GAME, Games, SCRIPTS, ScriptsAIM_3_All, LIBRARY_PATH, EnableScripts,\
#    LIBRARY_NAME, FRAMESKIP, ScriptsSF_9, RECORD, RENDER_SPEED, RenderMode,\
#    DEFAULT_RENDER_MODE, ScriptsSFC_9, ScriptsSF_3, ScriptsAIM_3, ScriptsSFC_3,\
#    ScriptsAIM_9, KeyMap, ALL_COMBINATIONS, SCRIPT_LENGTH, GAME_VERSION,\
#    RENDER_MODE, RenderSpeed

import cv2

class HumanAgent():
    def __init__(self, config, environment):
        self.config = config
        self.environment = environment
        self.config.print()
        # Current key has to be initialized before first input of keyboard
        self.key_to_action = CT.key_to_action[self.config.env.env_name]
        
    def train(self):
        pass
    
    def graph(self):
        pass
    
    def play(self):
        on_release = self.on_release
        on_press = self.on_press
        self.current_key = self.key_to_action['wait']
        # Start listening to the keyboard
        with Listener(on_press=on_press, on_release = on_release) as listener: 
            while True:
                self.environment.gym.render()
                action = self.current_key
                print("ACTIOOON", action)
                observation, reward, done = self.environment.act(action)
        
                if done == 1:
                    print("****************************************")
                    cv2.destroyAllWindows()
                    self.environment.new_game()
                    self.environment.gym.close()

        
                     
    def on_press(self, key):
        print("EEH", key)
        key = str(key).replace("'", "")  # Get keyboard input
    
        self.current_key = self.key_to_action[key]
#        if key in self.key_to_action.keys():
#            self.current_key = self.key_to_action[key]
    
    def on_release(self, key):
        pass#self.current_key = self.key_to_action[key]
    
        


