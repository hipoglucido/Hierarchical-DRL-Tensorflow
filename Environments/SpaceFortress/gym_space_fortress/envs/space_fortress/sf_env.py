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
import math

from enum import Enum
import time
import logging
from configuration import Constants as CT
import imageio
import shutil
import glob

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


class Panel:
    def __init__(self, height, font_path):
        self.length = 17
        history_keys = ['goals']#, 'actions']
        self.history = {}
        for k in history_keys:
            self.history[k] = ["..." for _ in range(self.length)]
        
        self.height = height
        self.width = 250
        
        self.history_limit_i = 50
        self.span = (self.height - self.history_limit_i) / self.length
        
        self.current_color = (0, 0, 0)
        self.old_color =  (150, 150, 150)
        
        self.font1 = ImageFont.truetype(font = font_path,
                                             size = 15)
        self.font2 = ImageFont.truetype(font = font_path,
                                             size = 25)
    def empty(self):
        for key in self.history:
            self.history[key] = []
        
    def add(self, key, item):
        item = item.replace('Key.', '')
        self.history[key][:-1] = self.history[key][1:]
        self.history[key][-1] = item
    
        
    def get_image(self, info):
        panel = Image.new("RGB", (self.width, self.height), "white")
        draw = ImageDraw.Draw(panel)
        j = 10
        for n, item in enumerate(self.history['goals']):
            #print(n, item)
            coords = (j, self.history_limit_i + self.span * n)
            if n == self.length - 1:
                color = self.current_color
                item = '> ' + item
            else:
                color = self.old_color
                if n == 0:
                    item = '...'
            draw.text(coords, item, color, font = self.font1)
      
        color = (150, 25, 25)
        draw.text((10, 10),"#%d" % info['steps'], color, font = self.font2)
        draw.text((int(self.width / 2), 3), "%d hits" % info['hits'],
                                          color, font = self.font1)
        draw.text((int(self.width / 2), 20), "%d lifes" % info['lifes'],
                                          color, font = self.font1)
        return panel
            
        

class SFEnv(gym.Env):
    # - Class variables, these are needed to communicate with the template in gym/core.py
    metadata = {'render.modes': ['rgb_array', 'human', 'minimal', 'terminal'],
                        'configure.required' : True}
    
      
    # Initialize the environment
    def __init__(self):
        self.imgs = []
        self.step_counter = 0
        self.goal_has_changed = False
        
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
        info = {}
        if self.env_name == 'SF-v0':
            info['fortress_hits'] = 0
        for frame in range(1):#self.config.env.action_repeat):
          
            if not isinstance(action, list):
                self.act(action)
            else:
                self.act(action[frame])
           
            self.update_logic()
            reward_delta = self.score() - self.config.env.time_penalty
            reward += reward_delta
            done = self.terminal_state()
            if self.env_name == 'SFC-v0':
                if reward == 1 - self.config.env.time_penalty:
                    done = 1
            elif self.env_name == 'SF-v0' and reward_delta:
                info['fortress_hits'] += 1
            if done:
                self.ep_counter += 1
        self.step_counter += 1
        self.ep_reward += reward
#        screen = np.ctypeslib.as_array(self.update_screen().contents)
       
        obs = np.ctypeslib.as_array(self.get_symbols().contents)
        original = obs.copy()
#        print('______________________')
#        for k in self.raw_features_name_to_ix:
#            print(k, obs[self.raw_features_name_to_ix[k]])
        
        self.scale_observation(obs)
        try:
            preprocessed_obs = self.preprocess_observation(obs)
        except:
            print("ORIGINAL", original)
            print("SCALA", obs)
            3/0
#        print("prep_obs", preprocessed_obs)
        
        #symbols = np.ctypeslib.as_array(self.get_symbols().contents)
#        ob = symbols
        #ob = symbols
        return preprocessed_obs, reward, done, info
    
    def scale_observation(self, raw_obs):
        if self.env_name == 'SFC-v0':
            
            #Normalize
            raw_obs[0] /= self.screen_height    # Ship_Y_Pos
            raw_obs[1] /= self.screen_width     # Ship_X_Pos
            raw_obs[2] /= 360                   # Ship_Headings
            raw_obs[3] /= self.screen_width     # Square_Y
            raw_obs[4] /= self.screen_height    # Square_X
            raw_obs[5] = raw_obs[5]                 # Square_steps
            raw_obs[6] = (raw_obs[6] + 5.) / 10.    # Ship_X_Speed
            raw_obs[7] = (raw_obs[7] + 5.) / 10.    # Ship_Y_Speed
        elif self.env_name == 'AIM-v0':
            #Normalize
            raw_obs[0] /= 360                   # Ship_Headings
            raw_obs[1] /= self.screen_width     # Mine_X_Pos
            raw_obs[2] /= self.screen_height    # Mine_Y_Pos
        elif self.env_name == 'SF-v0':
            raw_obs[0]  /= self.screen_height         # Ship_Y_Pos
            raw_obs[1]  /= self.screen_width          # Ship_X_Pos
            raw_obs[2]   = (raw_obs[2] + 5.) / 10.    # Ship_Y_Speed
            raw_obs[3]   = (raw_obs[3] + 5.) / 10.    # Ship_X_Speed 
            raw_obs[4]  /= 360                        # Ship_Headings
            raw_obs[5]  /= self.screen_height         # Missile_Y_Pos
            raw_obs[6]  /= self.screen_width          # Missile_X_Pos
            raw_obs[7]  /= 360                        # fort_Headings
            raw_obs[8] /= 100                        # Missile_Stock
    
            
    def preprocess_observation(self, obs):
        """
        symbols[0] = Ship_X_Pos;// /(float) WINDOW_WIDTH;	
        	symbols[1] = Ship_Y_Pos;// /(float) WINDOW_HEIGHT;
        	symbols[2] = Ship_Headings;// /(float) 360;
        	symbols[3] = Square_X;// /(float) WINDOW_WIDTH;
        	symbols[4] = Square_Y;// /(float) WINDOW_HEIGHT;
        	symbols[5] = Square_Step;//
        """
        
        features = []
        
        for prep_f in self.prep_fs:
#            print(prep_f)
            packed = prep_f(obs)
#            print(packed)
            for feature in packed:
#                print(feature)
                features.append(feature)
        preprocessed_obs = np.array(features)

        
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

        #print(type(img), img.mean(), img.max(), img.min())
        
            #image_name = self.env_name + "_" + self.current_time + ".png"
            #img_path = os.path.join(self.episode_dir, image_name)
            
    #
    #            if self.record_path is not None and self.config.env.record:
            #cv2.imwrite(img_path, img)
        env_img = Image.fromarray(img)      
        info = {'steps' : self.step_counter,
                'hits'  : self.get_vulner_counter(),
                'lifes' : self.get_lifes_remaining()}
        panel_img = self.panel.get_image(info)
        width = self.screen_width + self.panel.width
        height = self.screen_height
        full_image = Image.new('RGB', (width, height))
        full_image.paste(env_img, (0, 0))
        full_image.paste(panel_img, (self.screen_width, 0))
        self.imgs.append(full_image)
            #print(img.max(), img.min(),img.shape, type(img))
        
        if self.config.ag.agent_type == 'human' or self.config.gl.watch:
            cv2.imshow(self.env_name, np.array(full_image))
            cv2.waitKey(self.config.env.render_delay)
        else:
            if not os.path.exists(self.episode_dir):
                os.makedirs(self.episode_dir)
        
    def generate_video(self, delete_images = True):
        
#        img_paths = glob.glob(os.path.join(self.episode_dir, '*.png'))
#        img_paths.sort(key=os.path.getatime)
        if self.ep_reward is not None:
            video_path = self.episode_dir + "_R" + str(self.ep_reward)
        else:
            video_path = self.episode_dir
        video_path += '.mp4'
        perc = .7
   
        
        (original_width, original_heigth) = self.imgs[0].size
        #new_size = (int(perc * original_width), int(perc * original_heigth))
        
        #self.imgs = [img.resize(new_size, Image.ANTIALIAS) for img in self.imgs]
        self.imgs = [np.array(img) for img in self.imgs]
        blank = self.imgs[-1].copy() * 0 + 255
        self.imgs.append(blank)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, 20.0, (original_width, original_heigth))
        #imageio.mimsave(video_path, self.imgs, duration = .00001)
        for img in self.imgs:
            video.write(img)
        video.release()
        if delete_images:
            shutil.rmtree(self.episode_dir)
            

    def reset(self):
        self.window_active = False
        self.step_counter = 0
        self.reset_sf()
        if os.path.exists(self.episode_dir): 
            self.generate_video()
        self.imgs = []
        self.ep_reward = 0
        
        self.episode_dir = os.path.join(self.config.gl.logs_dir,
                                        self.config.model_name, 
                                        'episodes', "ep%d_%s" % \
                                        (self.ep_counter,self.current_time))
        
        obs = np.ctypeslib.as_array(self.get_symbols().contents)
#        print("obs:",obs)
        self.scale_observation(obs)
        preprocessed_obs = self.preprocess_observation(obs)
        return preprocessed_obs # For some reason should show the observation



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
        if self.is_no_direction:
            CT.key_to_action[self.env_name]['Key.down'] = 4
            CT.action_to_sf[self.env_name][4] = CT.key_to_sf['Key.down']
        self._action_set = CT.action_to_sf[self.env_name]
       
#        print(self._action_set)
#        print(CT.action_to_sf[self.env_name])
#        print(CT.key_to_action[self.env_name])
    def get_prep_feature(self, observation, feature_name):
        index = self.feature_names.index(feature_name)
        return observation[index]
        
    def get_raw_feature(self, observation, feature_name):
        return observation[self.raw_features_name_to_ix[feature_name]]
    def define_features(self):
        if self.env_name == 'SFC-v0':
            self.raw_features_name_to_ix = {
                    'ship_pos_i'   : 0,
                    'ship_pos_j'   : 1,
                    'ship_headings': 2,
                    'square_pos_i' : 3,
                    'square_pos_j' : 4,
                    #'square_steps' : 5,
                    'ship_speed_i' : 6,
                    'ship_speed_j' : 7
                    }
            
            
        elif self.env_name == 'AIM-v0':
            self.raw_features_name_to_ix = {
                    'ship_headings': 0,
                    'mine_pos_i'   : 1,
                    'mine_pos_j'   : 2
                    }
           
            
        elif self.env_name == 'SF-v0':
            self.raw_features_name_to_ix = {
                    'ship_pos_i'     : 0,
                    'ship_pos_j'     : 1,
                    'ship_speed_i'   : 2,
                    'ship_speed_j'   : 3,
                    'ship_headings'  : 4,
                    'missile_pos_i'  : 5,
                    'missile_pos_j'  : 6,
                    'fort_headings' : 7,
                    'missile_stock'  : 8
                    }
        
        prep_fs = []
        feature_names = [] 
      
        if self.env_name == 'AIM-v0':
            ##Irrelevant if WRAPPER / FRICTIONLESS
            prep_fs += [
                lambda obs: [self.get_raw_feature(obs, 'mine_pos_i')],
                lambda obs: [self.get_raw_feature(obs, 'mine_pos_j')]
            ]
            feature_names += ['mine_pos_i', 'mine_pos_j']
        elif not self.is_wrapper:
            # NOT WRAPPER
            if self.env_name == 'SFC-v0':
                prep_fs += [
                    lambda obs: [self.get_raw_feature(obs, 'ship_pos_i')],
                    lambda obs: [self.get_raw_feature(obs, 'ship_pos_j')],
                    lambda obs: [self.get_raw_feature(obs, 'square_pos_i')],
                    lambda obs: [self.get_raw_feature(obs, 'square_pos_j')]
                ]
                feature_names += ['ship_pos_i', 'ship_pos_j','square_pos_i', 'square_pos_j']
          
            elif self.env_name == 'SF-v0':
                prep_fs += [
                    lambda obs: [self.get_raw_feature(obs, 'ship_pos_i')],
                    lambda obs: [self.get_raw_feature(obs, 'ship_pos_j')],
                    lambda obs: [self.get_raw_feature(obs, 'missile_pos_i')],
                    lambda obs: [self.get_raw_feature(obs, 'missile_pos_j')],
                    lambda obs: [self.get_raw_feature(obs, 'missile_stock')]
                ]
                feature_names += ['ship_pos_i', 'ship_pos_j',
                                  'missile_pos_i', 'missile_pos_j',
                                  'missile_stock']
            
        else:
            # WRAPPER
            if self.env_name == 'SFC-v0':
                prep_fs += [
                    lambda obs: aux_decompose_cyclic(self.get_raw_feature(obs, 'ship_pos_i')),
                    lambda obs: aux_decompose_cyclic(self.get_raw_feature(obs, 'ship_pos_j')),
                    lambda obs: aux_decompose_cyclic(self.get_raw_feature(obs, 'square_pos_i')),
                    lambda obs: aux_decompose_cyclic(self.get_raw_feature(obs, 'square_pos_j'))
                ]
                aux = ['ship_pos_i', 'ship_pos_j', 'square_pos_i', 'square_pos_j']
                for fn in aux:
                    feature_names += [fn + '_sin', fn + '_cos']
                
          
            elif self.env_name == 'SF-v0':
                prep_fs += [
                    lambda obs: aux_decompose_cyclic(self.get_raw_feature(obs, 'ship_pos_i')),
                    lambda obs: aux_decompose_cyclic(self.get_raw_feature(obs, 'ship_pos_j'))
                ]
                aux = ['ship_pos_i', 'ship_pos_j']
                for fn in aux:
                    feature_names += [fn + '_sin', fn + '_cos']
                prep_fs += [
                    lambda obs: [self.get_raw_feature(obs, 'missile_pos_i')],
                    lambda obs: [self.get_raw_feature(obs, 'missile_pos_j')],
                    lambda obs: [self.get_raw_feature(obs, 'missile_stock')]
                ]
                feature_names += ['missile_pos_i', 'missile_pos_j',
                                  'missile_stock']
        if self.is_frictionless and self.env_name != 'AIM-v0':
                # FRICTIONLESS
                prep_fs += [
                    lambda obs: [self.get_raw_feature(obs, 'ship_speed_i')],
                    lambda obs: [self.get_raw_feature(obs, 'ship_speed_j')]
                    ]
                feature_names += ['ship_speed_i', 'ship_speed_j']          
                

        if self.is_no_direction:
            #Head doesn't control direction, no heading of the spaceship needed
            pass
        else:
            f = lambda obs: aux_decompose_cyclic(self.get_raw_feature(obs, 'ship_headings'))
            prep_fs.append(f)
            feature_names.append("ship_headings_sin")
            feature_names.append("ship_headings_cos")
            if self.env_name == 'SF-v0':
                pass
                #Fortress is ALWAYS looking at the spaceship, so this is not needed
#                f = lambda obs: aux_decompose_cyclic(self.get_raw_feature(obs, 'fort_headings'))
#                prep_fs.append(f)
#                feature_names.append("fort_headings_sin")
#                feature_names.append("fort_headings_cos")
        self.feature_names = feature_names
        self.state_size = len(self.feature_names)
        self.prep_fs = prep_fs
    def one_hot_inverse(self, screen):
        #TODO remove function adn adapt HDQN
        return None
    
        
        
    def configure(self, cnf):
        # Specify the game name which will be shown at the top of the game window
        
        self.config = cnf
        self.env_name = self.config.env.env_name
       
        
        self.logger = logging.getLogger()
  
        self.scale = 5.3
        
        libpath=self.config.env.library_path

        libname = self.env_name.split('-')[0].lower()


        libname += '_frame_lib'

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

        self.get_symbols = library.get_symbols
        n_raw_symbols = CT.SF_observation_space_sizes[self.env_name]
        self.get_symbols.restype = ctypes.POINTER(ctypes.c_float * n_raw_symbols)

        self.update_logic = library.SF_iteration
        self.update_screen = library.update_screen


        self.terminal_state = library.get_terminal_state
        self.score = library.get_score
        self.stop_drawing = library.stop_drawing
        self.pretty_screen = library.get_original_screen

        if self.env_name == 'SF-v0':
            self.get_vulner_counter = library.get_vulner_counter
            self.get_lifes_remaining = library.get_lifes_remaining
        sixteen_bit_img_bytes = self.screen_width * self.screen_height * 2
        self.pretty_screen.restype = ctypes.POINTER(ctypes.c_ubyte * sixteen_bit_img_bytes)
        self.score.restype = ctypes.c_float    
        
        # It is possible to specify a seed for random number generation
        self._seed()
        self.define_action_set()

        
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.state_space = gym.spaces.Discrete(self.state_size)

        self.init_game()


        self.ep_counter = 0
        self.episode_dir = os.path.join(self.config.gl.logs_dir,
                                        self.config.model_name, 
                                        'episodes', '_')
        
        print("Using features", ', '.join(self.feature_names))
        
       
        font_path = os.path.join(self.config.gl.others_dir, 'Consolas.ttf')
        self.panel = Panel(self.screen_height, font_path)
        

    
def aux_decompose_cyclic(x):
    """
    Decomposes a cyclic feature into x, y coordinates
    x : normalized feature (float, scalar)
    """
    try:
        assert 1 >= x >= 0
    except Exception as e:
        assert 1.1 >= x >= -0.1, "X must be normalized (%f)" % x
        x = np.clip(x, 0, 1)

    sin = math.sin(2 * math.pi * x)
    cos = math.cos(2 * math.pi * x)
    
    norm_sin = (sin + 1) / 2
    norm_cos = (cos + 1) / 2
    #return sin, cos
    return norm_sin, norm_cos
