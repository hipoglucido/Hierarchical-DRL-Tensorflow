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
            if self.env_name == 'SFC-v0':
                if reward == 1 - self.config.env.time_penalty:
                    done = 1
            if done:
                self.ep_counter += 1
        self.ep_reward += reward
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
        
        
        
        
        #Normalize
        obs[0] /= self.screen_width     # Ship_X_Pos
        obs[1] /= self.screen_height    # Ship_Y_Pos
        obs[2] /= 360              # Ship_Headings
        obs[3] /= self.screen_width     # Square_X
        obs[4] /= self.screen_height    # Square_Y
        
        features = []
        
        for prep_f in self.prep_fs:
#            print(prep_f)
            packed = prep_f(obs)
#            print(packed)
            for feature in packed:
#                print(feature)
                features.append(feature)
        preprocessed_obs = np.array(features)
#        feature_names = []
#        prep_obs = []
#        if not self.is_wrapper:
#            ship_pos_x, ship_pos_y = obs[0], obs[1]
#            square_pos_x, square_pos_y = obs[3], obs[4]
#            prep_obs += [ship_pos_x, ship_pos_y, square_pos_x, square_pos_y]
#            feature_names += ['ship_pos_x', 'ship_pos_y', 'square_pos_x', 'square_pos_y']
#        if self.env_name == 'SFC-v0':
#            if not self.is_wrapper:
#                
#                ship_headings_x, ship_headings_y = aux_decompose_cyclic(obs[2])
#                
#                #degrees = obs[2] * 360
#                preprocessed_obs = np.array([
#                        ship_pos_x,
#                        ship_pos_y,               
#                        ship_headings_x,
#                        ship_headings_y,
#                        square_pos_x,
#                        square_pos_y
#                        ])
#            else:                
#                ship_x_pos_x, ship_x_pos_y = aux_decompose_cyclic(obs[0])
#                ship_y_pos_x, ship_y_pos_y = aux_decompose_cyclic(obs[1])
#                ship_headings_x, ship_headings_y = aux_decompose_cyclic(obs[2])
#                square_x_pos_x, square_x_pos_y = aux_decompose_cyclic(obs[3])
#                square_y_pos_x, square_y_pos_y = aux_decompose_cyclic(obs[4])
#                
#                preprocessed_obs = np.array([
#                        ship_x_pos_x,
#                        ship_x_pos_y,                
#                        ship_y_pos_x,
#                        ship_y_pos_y,                
#                        ship_headings_x,
#                        ship_headings_y,
#                        square_x_pos_x,
#                        square_x_pos_y,
#                        square_y_pos_x,
#                        square_y_pos_y
#                        ])
#        else:
#            assert 0
        
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
            cv2.imshow(self.env_name, img)
            cv2.waitKey(self.config.env.render_delay)
        else:
            if not os.path.exists(self.episode_dir):
                os.makedirs(self.episode_dir)
            image_name = self.env_name + "_" + self.current_time + ".png"
            img_path = os.path.join(self.episode_dir, image_name)
            
    #
    #            if self.record_path is not None and self.config.env.record:
            cv2.imwrite(img_path, img)
            #print(img.max(), img.min(),img.shape, type(img))
        
        
        
    def generate_video(self, delete_images = True):
        imgs = []
        img_paths = glob.glob(os.path.join(self.episode_dir, '*.png'))
        img_paths.sort(key=os.path.getatime)
        if self.ep_reward is not None:
            video_path = self.episode_dir + "_R" + str(self.ep_reward)
        else:
            video_path = self.episode_dir
        video_path += '.gif'
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
        self.ep_reward = 0
        
        self.episode_dir = os.path.join(self.config.gl.logs_dir, self.config.model_dir, 
                                        'episodes', "ep%d_%s" % \
                                        (self.ep_counter,self.current_time))
        
        # screen = self.screen().contents
        # obv = np.ctypeslib.as_array(screen)
        return 0 # For some reason should show the observation



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
        cv2.namedWindow(self.env_name)
    # Configure the space fortress gym environment
    def define_action_set(self):
        self._action_set = CT.action_to_sf[self.env_name]
        if self.is_no_direction:
            self._action_set[3] = CT.key_to_sf['Key.down']
    def get_prep_feature(self, observation, feature_name):
        index = self.feature_names.index(feature_name)
        return observation[index]
        
    def get_raw_feature(self, observation, feature_name):
        #print("getting",feature_name)
        return observation[self.raw_features_name_to_ix[feature_name]]
    def define_features(self):
        self.raw_features_name_to_ix = {
                'ship_pos_i'   : 0,
                'ship_pos_j'   : 1,
                'ship_headings': 2,
                'square_pos_i' : 3,
                'square_pos_j' : 4
                }

        def aux_decompose_cyclic(x):
            """
            Decomposes a cyclic feature into x, y coordinates
            x : normalized feature (float, scalar)
            """
            try:
                assert 1 >= x >= 0, "X must be normalized (%f)" % x
            except Exception as e:
                assert 1.1 >= x >= -0.1
                x = np.clip(x, 0, 1)
            import math
            sin = math.sin(2 * math.pi * x)
            cos = math.cos(2 * math.pi * x)
            return sin, cos
        prep_fs = []
        feature_names = [] 
        coordinate_feature_names = ['ship_pos_i', 'ship_pos_j','square_pos_i', 'square_pos_j']
#        from copy import deepcopy
        if not self.is_wrapper:
#            for fn in coordinate_feature_names:
#                print(99,fn)
#                f = lambda obs: [self.get_raw_feature(obs, deepcopy(fn))]
#                prep_fs.append(f)
#                feature_names.append(fn)
            prep_fs = [
                lambda obs: [self.get_raw_feature(obs, 'ship_pos_i')],
                lambda obs: [self.get_raw_feature(obs, 'ship_pos_j')],
                lambda obs: [self.get_raw_feature(obs, 'square_pos_i')],
                lambda obs: [self.get_raw_feature(obs, 'square_pos_j')]
            ]
            feature_names += coordinate_feature_names
        else:
            for fn in coordinate_feature_names:
                f = lambda obs: aux_decompose_cyclic(self.get_raw_feature(obs, fn))
                prep_fs.append(f)
                feature_names.append(fn + "_sin")
                feature_names.append(fn + "_cos")
#            feature_names += ['ship_pos_i_x', 'ship_pos_i_y', 'ship_pos_j_x',
#                              'ship_pos_j_y','square_pos_i_x','square_pos_i_y',
#                              'square_pos_j_x','square_pos_j_y']
#            prep_fs += [
#                    ),
#                    lambda obs: aux_decompose_cyclic(self.get_raw_feature(obs, 'ship_pos_y')),
#                    lambda obs: aux_decompose_cyclic(self.get_raw_feature(obs, 'square_pos_x')),
#                    lambda obs: aux_decompose_cyclic(self.get_raw_feature(obs, 'square_pos_y'))
#                    ]
        if self.is_no_direction:
            #Head doesn't control direction, no heading of the spaceship needed
            pass
        else:
            fn = 'ship_headings'
            f = lambda obs: aux_decompose_cyclic(self.get_raw_feature(obs, fn))
            prep_fs.append(f)
            feature_names.append(fn + "_sin")
            feature_names.append(fn + "_cos")
        self.feature_names = feature_names
        self.state_size = len(self.feature_names)
        self.prep_fs = prep_fs 
    def configure(self, cnf):
        # Specify the game name which will be shown at the top of the game window
        
        self.config = cnf
        self.env_name = self.config.env.env_name
        
        self.logger = logging.getLogger()
        # The game which will be played, the possible games are
        # located in the enum Games in constants.py
         # The amount of (down) scaling of the screen height and width
        self.scale = 5.3
        
        libpath=self.config.env.library_path
#        print(LIBRARY_PATH)
        
        # Hard overwrite from constants.py
#        self.frame_skip = cnf.frameskip
        #self.mode = self.config.env.render_mode
        # Get the right shared library for the game
#        if self.env_name == 'SFS-v0':
#            libname = self.env_name
#        elif self.env_name =='AIM-v0' or self.env_name == \
#                            'SFC-v0' or self.env_name == 'SF-v0':
#            libname = self.game.lower()
#        else:
#            assert False
        libname = self.env_name.split('-')[0].lower()

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
        
        #self.update = library.update_frame
        self.init_game = library.start_drawing
        self.act = library.set_key
        self.reset_sf = library.reset_sf
        #self.screen = library.get_screen
        self.get_screen_width = library.get_screen_width
        self.get_screen_height = library.get_screen_height
        self.screen_height = self.get_screen_height()
        
        self.screen_width = self.get_screen_width()
        
        self.is_frictionless = library.is_frictionless()
        self.is_no_direction = library.is_no_direction()
        self.is_wrapper = library.is_wrapper()
    
        self.define_features()
#        if not self.is_frictionless and self.env_name == 'SFC-v0':
#        self.n_bytes =  ((int(self.screen_height/self.scale)) \
#                                        * (int(self.screen_width/self.scale)))
        
        
        self.get_symbols = library.get_symbols
        n_raw_symbols = CT.SF_observation_space_sizes[self.env_name]
        self.get_symbols.restype = ctypes.POINTER(ctypes.c_float * n_raw_symbols)
#        try:
        self.update_logic = ctypes.CDLL(lib_dir).SF_iteration
        self.update_screen = ctypes.CDLL(lib_dir).update_screen
#        self.update_screen.restype = ctypes.POINTER(ctypes.c_ubyte * self.n_bytes)
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
#        self.update.restype = ctypes.POINTER(ctypes.c_ubyte * self.n_bytes)
#        self.screen.restype = ctypes.POINTER(ctypes.c_ubyte * self.n_bytes)

        # 468 * 448 * 2 (original size times something to do with 16 bit images)
        sixteen_bit_img_bytes = self.screen_width * self.screen_height * 2
        self.pretty_screen.restype = ctypes.POINTER(ctypes.c_ubyte * sixteen_bit_img_bytes)
        self.score.restype = ctypes.c_float    
        
        # It is possible to specify a seed for random number generation
        self._seed()
        self.define_action_set()
#        if not self.is_frictionless() and self.env_name == 'SFC-v0':
#            del CT.action_to_sf[self.env_name][CT.key_to_sf['wait']]
#            action_set = {i : v for i, v in enumerate(CT.action_to_sf[self.env_name])}
#            self._action_set = action_set
#        else:
#            assert 0, str(self.is_frictionless()) + ', ' + self.env_name
#        print(self._action_set)
#        if self.env_name in ['SFS-v0','SF-v0']:
#            # All keys allowed
#            self._action_set = {0 : KeyMap.LEFT.value, 1 : KeyMap.UP.value, 2 : KeyMap.RIGHT.value, 3 : KeyMap.SHOOT.value}
#
#        elif self.env_name == 'AIM-v0':
#            # Only rotate left/right and shoot
#            self._action_set = {0 : KeyMap.SHOOT.value, 1 : KeyMap.LEFT.value, 2 : KeyMap.RIGHT.value}
#
#        elif self.env_name == 'SFC-v0':
#            # Only rotate left/right and forward
#            self._action_set = {0 : KeyMap.LEFT.value, 1 : KeyMap.RIGHT.value, 2 : KeyMap.UP.value}
#        else:
#            assert False
        
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.state_space = gym.spaces.Discrete(self.state_size)
        # 1 float = 4 bytes
        
        ###########
    


        # Initialize the game's drawing context and it's variables
        # I would rather that this be in the init method, but the OpenAI developer himself stated
        # that if some functionality of an enviroment depends on the render mode, the only way
        # to handle this is to write a configure method, a method that is only callable after the
        # init
        self.init_game()

       

        # add down movement when in no_direction mode

            
            
        self.ep_counter = 0
        self.episode_dir = os.path.join(self.config.gl.logs_dir, self.config.model_dir, 
                                        'episodes', '_')
        
        print("Using features", ', '.join(self.feature_names))
        # The size of the screen when playing in human mode
        

    

