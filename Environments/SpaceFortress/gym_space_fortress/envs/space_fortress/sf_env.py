import gym
from gym.utils import seeding
from gym import spaces
import ctypes
from time import sleep
from sys import platform
import datetime
import numpy as np
import cv2
import os
import csv
from pathlib import Path
import sys
#from constants import Games, SCRIPTS, ScriptsAIM_3_All, LIBRARY_PATH, EnableScripts,\
#    LIBRARY_NAME, FRAMESKIP, ScriptsSF_9, RECORD, RENDER_SPEED, RenderMode,\
#    DEFAULT_RENDER_MODE, ScriptsSFC_9, ScriptsSF_3, ScriptsAIM_3, ScriptsSFC_3,\
#    ScriptsAIM_9,  ALL_COMBINATIONS, SCRIPT_LENGTH
from enum import Enum
import time
import logging
from configuration import Constants as CT

# SFEnv is a child of the environment template located in gym/core.py
# This instance handles the space fortress environment
class SFEnv(gym.Env):
    # - Class variables, these are needed to communicate with the template in gym/core.py
    metadata = {'render.modes': ['rgb_array', 'human', 'minimal', 'terminal'],
                'configure.required' : True}
      
    # Initialize the environment
    def __init__(self):
        pass
        
        
    def _seed(self):
        #TODO
        pass
    @property
    # Returns the amount of actions
    def n_actions(self):
        return len(self._action_set)

    # Returns the best action
    def best_action(self):
        return self.best()

    def _step2(self, a):
        action = self._action_set[a] # Select the action from the action dict
        print(77777, action)
        self.act(action)
        ob = np.ctypeslib.as_array(self.update().contents)
        reward = self.score()
        ending = self.terminal_state()
        
        return ob, reward, ending, {}

    def step(self, a):
        action = self._action_set[a] # Select the action from the action dictq
        reward = 0.0
        done = False
        for frame in range(self.cnf.env.action_repeat):
            
            if not isinstance(action, list):
                self.act(action)
            else:
                self.act(action[frame])
            self.update_logic()
            reward += self.score()
            done = self.terminal_state()
            if done:
                break
        screen = np.ctypeslib.as_array(self.update_screen().contents)
        #symbols = np.ctypeslib.as_array(self.get_symbols().contents)
        
        #print("SYMBOLS:",symbols)
        
        #symbols = np.ctypeslib.as_array(self.get_symbols().contents)
        ob = screen
        #ob = symbols
        return ob, reward, done, {}

    # Renders the current state of the game, only for our visualisation purposes
    # it is not important for the learning algorithm
    def render(self):
        
        if not self.cnf.env.render_mode == "rgb_array":
#            img = None
#            render_delay = None
#            new_frame=None

            if self.cnf.env.render_mode == 'human':
                new_frame = self.pretty_screen().contents
            else:
                new_frame = self.screen().contents
            img = np.ctypeslib.as_array(new_frame)

            if self.cnf.env.render_mode =='human':
                img = np.reshape(img, (self.screen_height, self.screen_width, 2))
                img = cv2.cvtColor(img, cv2.COLOR_BGR5652RGB)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif self.cnf.env.render_mode == 'minimal':
                img = np.reshape(img, (int(self.screen_height/self.scale),
                                       int(self.screen_width/self.scale)))

            

            if self.record_path is not None and self.cnf.env.record:
                current_time = str(datetime.datetime.now().time().isoformat())\
                                                        .replace("/", ":")
                cv2.imwrite(self.record_path + "/sf" + current_time + ".png", img)

            cv2.imshow(self.game_name, img)
            cv2.waitKey(self.cnf.env.render_delay)



    def reset(self):
        self.reset_sf()
        # screen = self.screen().contents
        # obv = np.ctypeslib.as_array(screen)
        return 0 # For some reason should show the observation


    def write_out_stats(self , file_id=None):
        current_time = str(datetime.datetime.now().time().isoformat())\
                                                            .replace("/", ":")
        id = file_id if file_id else current_time
        SHIP_WON = 1 # some constant from the c interface
        keys = ["Won"]
        with open(os.path.join('gym_stats', self.game_name+"-"+id+'.csv'),
                                                              'wb') as csvfile:
            dict_writer = csv.DictWriter(csvfile, fieldnames=keys)
            dict_writer.writeheader()
            for t in self.terminal_states:
                dict_writer.writerow({"Won" : t == 1})

        self.terminal_states = []
        csvfile.close()

    def close(self):
#        if self.write_stats:
#            self.write_out_stats()
        # maybe condition the stats?
#        self.write_out_stats()
        self.stop_drawing()
        sys.exit(0)

    # Configure the space fortress gym environment
    def configure(self, cnf):
        # Specify the game name which will be shown at the top of the game window
        
        self.cnf = cnf
        self.game_name = self.cnf.env.env_name
        
        self.logger = logging.getLogger()
        # The game which will be played, the possible games are
        # located in the enum Games in constants.py
        
        # The size of the screen when playing in human mode
        self.screen_height = 448
        self.screen_width = 448
         # The amount of (down) scaling of the screen height and width
        self.scale = 5.3
        # It is possible to specify a seed for random number generation
        self._seed()
        self._action_set = CT.action_to_sf[self.game_name]
#        if self.game_name in ['SFS-v0','SF-v0']:
#            # All keys allowed
#            self._action_set = {0 : KeyMap.LEFT.value, 1 : KeyMap.UP.value, 2 : KeyMap.RIGHT.value, 3 : KeyMap.SHOOT.value}
#
#        elif self.game_name == 'AIM-v0':
#            # Only rotate left/right and shoot
#            self._action_set = {0 : KeyMap.SHOOT.value, 1 : KeyMap.LEFT.value, 2 : KeyMap.RIGHT.value}
#
#        elif self.game_name == 'SFC-v0':
#            # Only rotate left/right and forward
#            self._action_set = {0 : KeyMap.LEFT.value, 1 : KeyMap.RIGHT.value, 2 : KeyMap.UP.value}
#        else:
#            assert False
        self.state_size = 6
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.state_space = gym.spaces.Discrete(self.state_size)
        # 1 float = 4 bytes
        self.n_bytes = self.state_size * 4#((int(self.screen_height/self.scale)) * (int(self.screen_width/self.scale)))
        
        ###########
    
        debug=False
        record_path=None,
        no_direction=False
        lib_suffix=""
        
        libpath=self.cnf.env.library_path
#        print(LIBRARY_PATH)
        self.debug = debug
        # Hard overwrite from constants.py
#        self.frame_skip = cnf.frameskip
        #self.mode = self.cnf.env.render_mode
        # Get the right shared library for the game
#        if self.game_name == 'SFS-v0':
#            libname = self.game_name
#        elif self.game_name =='AIM-v0' or self.game_name == \
#                            'SFC-v0' or self.game_name == 'SF-v0':
#            libname = self.game.lower()
#        else:
#            assert False
        libname = self.game_name.split('-')[0].lower()

        # There is no need for a window when in RGB_ARRAY mode
        if self.cnf.env.render_mode != "rgb_array":
            cv2.namedWindow(self.game_name)

        libname += '_frame_lib'
        if self.cnf.env.render_mode == 'human':
            libname += "_FULL"

        libname += ".so"

        
        self.logger.info("With FrameSkip: %s" % self.cnf.env.action_repeat)

        # Link the environment to the shared libraries
        lib_dir = os.path.join(libpath, libname)
        print(lib_dir)
        library = ctypes.CDLL(lib_dir)
        self.logger.info("LOAD "+ lib_dir)
        
        self.update = library.update_frame
        self.init_game = library.start_drawing
        self.act = library.set_key
        self.reset_sf = library.reset_sf
        self.screen = library.get_screen
        self.get_symbols = library.get_symbols
        self.get_symbols.restype = ctypes.POINTER(ctypes.c_float * 6)
        try:
            self.update_logic = ctypes.CDLL(lib_dir).SF_iteration
            self.update_screen = ctypes.CDLL(lib_dir).update_screen
            self.update_screen.restype = ctypes.POINTER(ctypes.c_ubyte * self.n_bytes)
        except:
            print("Warning: Some functions where not found in the library.")
        try:
            self.best = ctypes.CDLL(lib_dir).get_best_move
        except: # Not implemented in the game yet
            print("Warning: best_move function not found in the library.")

        self.terminal_state = ctypes.CDLL(libpath +'/'+libname).get_terminal_state
        self.score = ctypes.CDLL(libpath +'/'+libname).get_score
        self.stop_drawing = ctypes.CDLL(libpath +'/'+libname).stop_drawing
        self.pretty_screen = ctypes.CDLL(libpath +'/'+libname).get_original_screen
        # Configure how many bytes to read in from the pointer
        # c_ubyte is equal to unsigned char
        self.update.restype = ctypes.POINTER(ctypes.c_ubyte * self.n_bytes)
        self.screen.restype = ctypes.POINTER(ctypes.c_ubyte * self.n_bytes)

        # 468 * 448 * 2 (original size times something to do with 16 bit images)
        sixteen_bit_img_bytes = self.screen_width * self.screen_height * 2
        self.pretty_screen.restype = ctypes.POINTER(ctypes.c_ubyte * sixteen_bit_img_bytes)
        self.score.restype = ctypes.c_float

        # Initialize the game's drawing context and it's variables
        # I would rather that this be in the init method, but the OpenAI developer himself stated
        # that if some functionality of an enviroment depends on the render mode, the only way
        # to handle this is to write a configure method, a method that is only callable after the
        # init
        self.init_game()

        self.record_path = record_path

        # add down movement when in no_direction mode
        if no_direction:
            self._action_set[3] = 65364
        
        

