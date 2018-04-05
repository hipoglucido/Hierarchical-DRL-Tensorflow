#!/usr/bin/env python
import gym
import sys
import numpy as np
from random import random
import cv2 # remove at one point
from time import sleep
from pynput.keyboard import KeyCode, Key, Listener
from constants import GAME, Games, SCRIPTS, ScriptsAIM_3_All, LIBRARY_PATH, EnableScripts,\
    LIBRARY_NAME, FRAMESKIP, ScriptsSF_9, RECORD, RENDER_SPEED, RenderMode,\
    DEFAULT_RENDER_MODE, ScriptsSFC_9, ScriptsSF_3, ScriptsAIM_3, ScriptsSFC_3,\
    ScriptsAIM_9, KeyMap, ALL_COMBINATIONS, SCRIPT_LENGTH, GAME_VERSION,\
    RENDER_MODE, RenderSpeed

# Specify the game the gym environment will play.
# All games are registered in gym/envs/__init__.py
# All possible versions of space fortress are located in constants.py

game_name = GAME.value + "-" + GAME_VERSION
env = gym.make(game_name)

# Configure enviroment

# Get the length of the scripts
if SCRIPTS.value == "on":
    script_length = SCRIPT_LENGTH.value 
else:
    script_length = 1

env.configure(mode=RENDER_MODE.value, record_path=None, no_direction=False,
              frame_skip=script_length)


def on_press(key):
    key = str(key).replace("'", "")  # Get keyboard input
    global current_key               # Global var to use outside function
    
    
    # Keymap input to action space using dictionary
    if SCRIPTS.value == "on": # Bind buttons to use correct scripts
        if GAME.value == "SF" or GAME.value == "SFS":
            key_to_action = {"uz" : 0, "ux" : 1, "uc" : 2, "uv" : 3, "ub" : 4} 

        elif GAME.value == "SFC":
            key_to_action = {"uz" : 0, "ux" : 1, "uc" : 2, "uv" : 3, "ub" : 4,
                             "un" : 5, "um" : 6} 
            
        elif GAME.value == "AIM":
            key_to_action = {"uz" : 0, "ux" : 1, "uc" : 2, "uv" : 3, "ub" : 4,
                             "un" : 5, "um" : 6} 
            
    else:                     # Otherwise use normal actions with arrow keys
        if GAME.value == "SFC":
            key_to_action = {"Key.left" : 0, "Key.up" : 2, "Key.right" : 1} 
            
        elif GAME.value == "SF" or GAME.value == "SFS":
            key_to_action = {"Key.left" : 0, "Key.up" : 1, "Key.right" : 2,
                             "Key.down" : 3, "Key.space" : 3} 
            
        elif GAME.value == "AIM":
            key_to_action = {"Key.left" : 1, "Key.up" : 0, "Key.right" : 2,
                             "Key.down" : 0, "Key.space" : 0} 
        
    if key in key_to_action.keys():
        current_key = key_to_action[key]

# Do nothing on release 
def on_release(key):
    pass

# Current key has to be initialized before first input of keyboard
current_key = 0

def play():
    # Start listening to the keyboard
    with Listener(on_press=on_press, on_release=on_release) as listener:
        # Stop listening if sample actions should be used instead of keyboard
        if RENDER_SPEED != RenderSpeed.DEBUG:
            listener.stop()
            
        while True:
            env.render(mode=RENDER_MODE.value)
            if RENDER_SPEED == RenderSpeed.DEBUG:
                action = current_key
                
            else:
                action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)
            
         
            
            if done == 1:
                env.reset()

play()
