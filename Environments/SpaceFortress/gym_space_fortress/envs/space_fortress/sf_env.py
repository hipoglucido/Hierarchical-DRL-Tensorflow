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


from enum import Enum
import time
import logging
from configuration import Constants as CT
import imageio
import shutil
import glob
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



    def step(self, a):
        action = self._action_set[a] # Select the action from the action dictq
        reward = 0.0
        done = False
        for frame in range(self.config.env.action_repeat):
          
            if not isinstance(action, list):
                self.act(action)
            else:
                self.act(action[frame])
            self.update_logic()
            reward += self.score() - self.config.env.time_penalty
            done = self.terminal_state()
            if self.game_name == 'SFC-v0':
                if reward == 1 - self.config.env.time_penalty:
                    done = 1
            if done:
                self.ep_counter += 1
        
#        screen = np.ctypeslib.as_array(self.update_screen().contents)
        obs = np.ctypeslib.as_array(self.get_symbols().contents)
#        print("obs:",obs)
        preprocessed_obs = self.preprocess_observation(obs)
#        print("prep_obs", preprocessed_obs)
        
        #symbols = np.ctypeslib.as_array(self.get_symbols().contents)
#        ob = symbols
        #ob = symbols
        return preprocessed_obs, reward, done, {}
    def preprocess_observation(self, obs):
        """
        symbols[0] = Ship_X_Pos;// /(float) WINDOW_WIDTH;	
        	symbols[1] = Ship_Y_Pos;// /(float) WINDOW_HEIGHT;
        	symbols[2] = Ship_Headings;// /(float) 360;
        	symbols[3] = Square_X;// /(float) WINDOW_WIDTH;
        	symbols[4] = Square_Y;// /(float) WINDOW_HEIGHT;
        	symbols[5] = Square_Step;//
        """
        def aux_decompose_cyclic(x):
            """
            Decomposes a cyclic feature into x, y coordinates
            x : normalized feature (float, scalar)
            """
            assert 1 >= x >= 0, "X must be normalized (%f)" % x
            import math
            x = math.sin(2 * math.pi * x)
            y = math.cos(2 * math.pi * x)
            return x, y
        
        
        
        #Normalize
        obs[0] /= self.screen_width     # Ship_X_Pos
        obs[1] /= self.screen_height    # Ship_Y_Pos
        obs[2] /= 360              # Ship_Headings
        obs[3] /= self.screen_width     # Square_X
        obs[4] /= self.screen_height    # Square_Y
        
        ship_x_pos_x, ship_x_pos_y = aux_decompose_cyclic(obs[0])
        ship_y_pos_x, ship_y_pos_y = aux_decompose_cyclic(obs[1])
        ship_headings_x, ship_headings_y = aux_decompose_cyclic(obs[2])
        square_x_pos_x, square_x_pos_y = aux_decompose_cyclic(obs[3])
        square_y_pos_x, square_y_pos_y = aux_decompose_cyclic(obs[4])
        
        preprocessed_obs = np.array([
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
                ])
        return preprocessed_obs
    @property
    def current_time(self):
        return str(datetime.datetime.now().time().isoformat())\
                                                .replace("/", ":")
    # Renders the current state of the game, only for our visualisation purposes
    # it is not important for the learning algorithm
    def render(self):
        if not self.window_active and self.config.ag == 'human':
            self.open_window()
            self.window_active = True
        self.update_screen()
        
#    if not self.config.env.render_mode == "rgb_array":
#            img = None
#            render_delay = None
#            new_frame=None

#            if self.config.env.render_mode == 'human':
        new_frame = self.pretty_screen().contents
#            else:
#                new_frame = self.screen().contents
        img = np.ctypeslib.as_array(new_frame)

#            if self.config.env.render_mode =='human':
        img = np.reshape(img, (self.screen_height, self.screen_width, 2))
        img = cv2.cvtColor(img, cv2.COLOR_BGR5652RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#            elif self.config.env.render_mode == 'minimal':
#                img = np.reshape(img, (int(self.screen_height/self.scale),
#                                       int(self.screen_width/self.scale)))

   
        if self.config.ag.agent_type == 'human':
            cv2.imshow(self.game_name, img)
            cv2.waitKey(self.config.env.render_delay)
        else:
            if not os.path.exists(self.episode_dir):
                os.makedirs(self.episode_dir)
            image_name = self.game_name + "_" + self.current_time + ".png"
            img_path = os.path.join(self.episode_dir, image_name)
            
    #
    #            if self.record_path is not None and self.config.env.record:
            cv2.imwrite(img_path, img)
            #print(img.max(), img.min(),img.shape, type(img))
        
        
        
    def generate_video(self, delete_images = True):
        imgs = []
        img_paths = glob.glob(os.path.join(self.episode_dir, '*.png'))
        img_paths.sort(key=os.path.getatime)
        video_path = self.episode_dir + '.gif'
        for img_path in img_paths:
            img = imageio.imread(img_path)
            imgs.append(img)
        blank = img.copy() * 0 + 255
        imgs.append(blank)
        imageio.mimsave(video_path, imgs, duration = .00001)
        if delete_images:
            shutil.rmtree(self.episode_dir)

    def reset(self):
        self.window_active = False
        self.reset_sf()
        if os.path.exists(self.episode_dir):
            
            self.generate_video()
        
        self.episode_dir = os.path.join(self.config.gl.logs_dir, self.config.model_dir, 
                                        'episodes', "ep%d_%s" % \
                                        (self.ep_counter,self.current_time))
        
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
        print("Closing!")
        self.stop_drawing()
        sys.exit(0)
    def open_window(self):
        print("Opening window!")
        cv2.namedWindow(self.game_name)
    # Configure the space fortress gym environment
    def configure(self, cnf):
        # Specify the game name which will be shown at the top of the game window
        
        self.config = cnf
        self.game_name = self.config.env.env_name
        
        self.logger = logging.getLogger()
        # The game which will be played, the possible games are
        # located in the enum Games in constants.py
        
        
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
        self.state_size = CT.SF_observation_space_sizes[self.game_name]
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.state_space = gym.spaces.Discrete(self.state_size)
        # 1 float = 4 bytes
        
        ###########
    
        debug=False
        record_path=None,
        no_direction=False
        lib_suffix=""
        
        libpath=self.config.env.library_path
#        print(LIBRARY_PATH)
        self.debug = debug
        # Hard overwrite from constants.py
#        self.frame_skip = cnf.frameskip
        #self.mode = self.config.env.render_mode
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
#        if self.config.env.render_mode != "rgb_array":
#            self.open_window()

        libname += '_frame_lib'
#        if self.config.env.render_mode == 'human':
#            libname += "_FULL"

        libname += ".so"

        
        self.logger.info("With FrameSkip: %s" % self.config.env.action_repeat)

        # Link the environment to the shared libraries
        lib_dir = os.path.join(libpath, libname)
#        print(lib_dir)
        library = ctypes.CDLL(lib_dir)
        self.logger.info("LOAD "+ lib_dir)
        
        self.update = library.update_frame
        self.init_game = library.start_drawing
        self.act = library.set_key
        self.reset_sf = library.reset_sf
        self.screen = library.get_screen
        self.get_screen_width = library.get_screen_width
        self.get_screen_height = library.get_screen_height
        self.screen_height = self.get_screen_height()
        
        self.screen_width = self.get_screen_width()
        self.n_bytes = self.state_size * ((int(self.screen_height/self.scale)) * (int(self.screen_width/self.scale)))
        
        self.get_symbols = library.get_symbols
        self.get_symbols.restype = ctypes.POINTER(ctypes.c_float * self.state_size)
#        try:
        self.update_logic = ctypes.CDLL(lib_dir).SF_iteration
        self.update_screen = ctypes.CDLL(lib_dir).update_screen
        self.update_screen.restype = ctypes.POINTER(ctypes.c_ubyte * self.n_bytes)
#        except:
#            print("Warning: Some functions where not found in the library.")
#        try:
#            self.best = ctypes.CDLL(lib_dir).get_best_move
#        except: # Not implemented in the game yet
#            print("Warning: best_move function not found in the library.")

        self.terminal_state = ctypes.CDLL(lib_dir).get_terminal_state
        self.score = ctypes.CDLL(lib_dir).get_score
        self.stop_drawing = ctypes.CDLL(lib_dir).stop_drawing
        self.pretty_screen = ctypes.CDLL(lib_dir).get_original_screen
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
            
            
        self.ep_counter = 0
        self.episode_dir = os.path.join(self.config.gl.logs_dir, self.config.model_dir, 
                                        'episodes', '_')
        
        # The size of the screen when playing in human mode
        
        
        

