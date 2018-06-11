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
from pathlib import Path
import sys
import math

from enum import Enum
import time
import logging
from constants import Constants as CT
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
    """
    This class only serves for visualization purposes. It appends an image
    to each frame with information about goals, action, rewards etc.
    """
    def __init__(self, height, font_path):
        self.length = 17
        self.history_keys = ['goals', 'actions', 'rewards']
        self.reset()
        
        self.height = height
        self.width = 315
        
        self.history_limit_i = 50
        self.span = (self.height - self.history_limit_i) / self.length
        
        self.current_color = (0, 0, 0)
        self.old_color =  (150, 150, 150)
        
        self.font1 = ImageFont.truetype(font = font_path,
                                             size = 15)
        self.font2 = ImageFont.truetype(font = font_path,
                                             size = 25)
    def reset(self):
        self.history = {}
        for k in self.history_keys:
            self.history[k] = ["..." for _ in range(self.length)]
        
    def add(self, key, item):
        if not isinstance(item, str) and item % 1 == 0:
            item = int(item)
        item = str(item).replace('Key.', '')
        self.history[key][:-1] = self.history[key][1:]
        self.history[key][-1] = item
    
        
    def get_image(self, info):
        panel = Image.new("RGB", (self.width, self.height), "white")
        draw = ImageDraw.Draw(panel)
        #Draw goals
        j = int(self.width * .04)
        for n, item in enumerate(self.history['goals']):
            #print(n, item)
            coords = (j, self.history_limit_i + self.span * n)
            if n == self.length - 1:
                color = self.current_color
                item = '> ' + item
            else:
                color = self.old_color
                if n == 0:
                    item = 'Goals:'
            draw.text(coords, item, color, font = self.font1)
        #Draw rewards
        j = int(self.width * .6)
        for n, item in enumerate(self.history['rewards']):
            #print(n, item)
            coords = (j, self.history_limit_i + self.span * n)
            if n == self.length - 1:
                color = self.current_color
                item = '> ' + item
            else:
                color = self.old_color
                if n == 0:
                    item = 'R:'
            draw.text(coords, item, color, font = self.font1)
        
        #Draw actions
        j = int(self.width * .7)
        for n, item in enumerate(self.history['actions']):
            #print(n, item)
            coords = (j, self.history_limit_i + self.span * n)
            if n == self.length - 1:
                color = self.current_color
                item = '> ' + item
            else:
                color = self.old_color
                if n == 0:
                    item = 'Actions:'
            draw.text(coords, item, color, font = self.font1)
      
        color = (150, 25, 25)
        j = int(self.width * .7)
        draw.text((10, 10),"#%d" % info['steps'], color, font = self.font2)
        draw.text((j, 3), "%s debug" % info['ship'],
                                          color, font = self.font1)
        draw.text((j, 20), "%d fortress" % info['fortress'],
                                          color, font = self.font1)
        return panel
            
        

class SFEnv(gym.Env):
    """
    Space Fortress Gym
    """
    
  
    def __init__(self):
        self.imgs = []
        self.step_counter = 0
        self.ep_counter = 0
        self.ep_reward = 0
        self.steps_since_last_shot = 1e10
        self.steps_since_last_fortress_hit = 1e10
        self.goal_has_changed = False
        self.print_shit = False
        self.currently_wrapping = False
        self.penalize_wrapping = False
        self.win = False
        
        #Just for metrics
        self.shot_too_fast = False
        
        self.delete = ''
        
    
        
    def _seed(self):
        #TODO
        pass
    
    @property
    # Returns the amount of actions
    def n_actions(self):
        return len(self._action_set)


    def after_episode(self):
        self.ep_counter += 1
        if len(self.imgs) > 0 and \
            (self.step_counter <= self.config.env.max_loops + 1 or \
                                     self.config.ag.mode == 'play'):
            self.generate_video()
            
    def get_custom_reward(self, action):
        reward = 0        
        # Penalize shooting fast
        if self.is_shot(action) and \
                    self.steps_since_last_shot < \
                                self.config.env.min_steps_between_shots and \
                    self.fortress_lifes > 2:
            reward -= self.config.env.fast_shooting_penalty
            self.shot_too_fast = True
        else:
            self.shot_too_fast = False
            
        if self.is_shot(action):
            self.steps_since_last_shot = 0
        else:
            self.steps_since_last_shot += 1

        if self.did_I_hit_mine() and self.config.env.mines_activated:
            reward += 1
            
            
        if self.did_I_hit_fortress() and self.step_counter != 0:
            
            self.fortress_lifes -= 1
            if self.fortress_lifes == 0 and \
                    self.steps_since_last_fortress_hit < \
                            self.config.env.min_steps_between_fortress_hits:
                reward = self.config.env.final_double_shot_reward 
                # WIN!
                self.win = True
            elif self.steps_since_last_fortress_hit > \
                            self.config.env.min_steps_between_fortress_hits:
                reward += 1
                
            else:
                # You shoot too fast when it was not allowed
                self.fortress_lifes += 1           
            self.steps_since_last_fortress_hit = 0
        else:            
            # Didn't hit the fortress
            self.steps_since_last_fortress_hit += 1
            if self.steps_since_last_fortress_hit > \
                        self.config.env.min_steps_between_fortress_hits and \
                                        self.fortress_lifes == 1:
                # fortress_lifes 1 -> 2
                pass#self.fortress_lifes = 2
            
    
            
        # Bad
        if self.did_mine_hit_me() and self.config.env.mines_activated:
            reward -= 1
            self.ship_lifes -= 1
        if self.did_fortress_hit_me():
            reward -= 1
            self.ship_lifes -= 1
        
        # Time penalty
        reward -= self.config.env.time_penalty
        
        # Penalize wrapping
        if self.penalize_wrapping:
            reward -= 1
        return reward
    
        
    def is_shot(self, action):
        return action == CT.key_to_action[self.env_name]['Key.space']
        
    def perform_action(self, action):
        key_id = self._action_set[action]
        self.set_key(key_id)        
        self.SF_iteration()

    def is_terminal(self):
        is_terminal = False
        if self.win:
            is_terminal = True
        elif self.fortress_lifes == 0:
            is_terminal = True
        elif self.step_counter >= self.config.env.max_loops:
            is_terminal = True        
        return is_terminal
    
    def step(self, action):
        """
        Performs one action on the environment. Always atomic actions 
        (granularity = 1). This method is agnostic to the
        action_repeat / frameskip that is being used by the agent.
        """
        
        #Call the C++ function
        self.perform_action(action)       
        reward = self.get_custom_reward(action)

        done = self.is_terminal()
        destroyed = int(self.fortress_lifes == 0)
        info = {'fortress_hit'               : self.did_I_hit_fortress(),
                'mine_hit'                   : self.did_I_hit_mine(),
                'win'                        : int(self.win),
                'destroyed'                  : destroyed,
                'steps'                      : self.step_counter, 
                'wrap_penalization'          : int(self.penalize_wrapping),
                'shot_too_fast_penalization' : int(self.shot_too_fast)}
        
        
        self.restart_variables()
        self.step_counter += 1
        self.ep_reward += reward
        observation = self.get_observation()
        
        #info['steps'] = 
        self.delete =str(self.get_prep_feature(observation, 'fortress_lifes'))
     
        return observation, reward, done, info
    
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
            raw_obs[8]  /= 100                        # Missile_Stock
            raw_obs[9]  /= self.screen_height         # Mine_Y_Pos
            raw_obs[10] /= self.screen_width          # Mine_X_Pos
            raw_obs[11] /= self.config.env.fortress_lifes
            raw_obs[12] = np.tanh(raw_obs[12] * .1)#1 / (1 + np.exp(-raw_obs[12]))
 
#            feature_names += ['mine_pos_i', 'mine_pos_j', 'fortress_lifes',
#                              'steps_since_last_shot']
            
            # If missiles are away from the screen, put them as 0, 0
            if self.last_shell_coords == (raw_obs[5], raw_obs[6]):
                raw_obs[5], raw_obs[6] = 0., 0.
            else:
                self.last_shell_coords = (raw_obs[5], raw_obs[6])
            if self.last_mine_coords == (raw_obs[9], raw_obs[10]):
                raw_obs[9], raw_obs[10] = 0., 0.
            else:
                self.last_mine_coords = (raw_obs[9], raw_obs[10])
            
        raw_obs = np.clip(raw_obs, 0, 1)
        """
            for i, obs in enumerate(raw_obs):
                if not 0 <= obs <= 1 and i not in [5, 6, 8]:
                    print(i, obs)
        """    
        return raw_obs    
    
            
    def preprocess_observation(self, obs):
        """
        Takes one scaled environment state and applies some preprocessing. Note
        that the resulting vector may have different dimension that the input
        
        params:
            obs: array of floats
        returns:
            preprocessed_obs: array of floatss
        """        
        features = []
        for prep_f in self.prep_fs:
            packed = prep_f(obs)
            for feature in packed:
                features.append(feature)
        preprocessed_obs = np.array(features)
        return preprocessed_obs
    
    @property
    def current_time(self):
        return str(datetime.datetime.now().time().isoformat())\
                                                .replace("/", ":")
    
    def render(self):
        """
        Renders the current state of the game, only for our visualisation
        purposes it is not important for the learning algorithm. Visualization
        means either through a window or just for making a video file from the
        current episode that will be written to disk (and not visualized in a 
        window)
        """
        
        if not self.window_active and self.config.ag.agent_type == 'human' or \
                                                self.config.gl.watch:
            #Opens a window if is needed
            self.open_window()
            self.window_active = True
        
        #Calls C++ function to update screen content
        self.update_screen()
        
        #Build the image out of that
        new_frame = self.pretty_screen().contents
        img = np.ctypeslib.as_array(new_frame)
        img = np.reshape(img, (self.screen_height, self.screen_width, 2))
        img = cv2.cvtColor(img, cv2.COLOR_BGR5652RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Adds the panel to right of the image
        env_img = Image.fromarray(img)      
        info = {'steps'     : self.step_counter,
                'fortress'  : self.fortress_lifes,
                'ship'      : self.delete}#self.ship_lifes}
        panel_img = self.panel.get_image(info)
        width = self.screen_width + self.panel.width
        height = self.screen_height
        full_image = Image.new('RGB', (width, height))
        full_image.paste(env_img, (0, 0))
        full_image.paste(panel_img, (self.screen_width, 0))
        
        #Appends the full image to the current episode's list of images
        self.imgs.append(full_image)

        if self.window_active:
            # Displays image on a window if this is opened
            cv2.imshow(self.env_name, np.array(full_image))
            cv2.waitKey(self.config.env.render_delay)
        
            
        
    def generate_video(self):
        """
        Takes the list of images of the current episode and makes a video out
        of them
        """
        
        video_name = "ep%d_%s_R%d_win%d.mp4" % (self.ep_counter, self.current_time, \
                                                              self.ep_reward, int(self.win))
        video_path = os.path.join(self.episode_dir, video_name)
        (original_width, original_heigth) = self.imgs[0].size
        # PIL image >>> np.array
        self.imgs = [np.array(img) for img in self.imgs]
        #Adds a white frame at the end (useful if .gif is produced)
        last_frame = self.imgs[-1].copy()
        if self.win:
            # Debug and see reward
            self.imgs.append(last_frame)
        blank = last_frame * 0 + 255
        self.imgs.append(blank)
        #Make video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, 20.0, (original_width, original_heigth))
        #imageio.mimsave(video_path, self.imgs, duration = .00001)
        for img in self.imgs:
            video.write(img)
        video.release()
    
    def check_wrapping(self, obs):
        """
        Checks if the spaceship is wrapping. If yes, that should be penalized
        """
        i = self.get_raw_feature(obs, 'ship_pos_i')
        j = self.get_raw_feature(obs, 'ship_pos_j')
        eps = 0.02
        if 0 <= i < eps or \
               0 <= j < eps or \
               1 - eps < i <= 1 or \
               1 - eps < j <= 1:
            if not self.currently_wrapping:
                self.penalize_wrapping = True
            else:
                self.penalize_wrapping = False
            self.currently_wrapping = True
        else:
            self.penalize_wrapping = False
            self.currently_wrapping = False
    def generate_extra_features(self, observation):
        pass
        
    def get_raw_observation(self):  
        # From C++ game
        game_obs = np.ctypeslib.as_array(self.get_symbols().contents)
        
        # Extra features
        extra_obs = np.array([self.fortress_lifes, self.steps_since_last_shot], dtype = np.float)
        
        # Concat
        raw_observation = np.hstack([game_obs, extra_obs])
        return raw_observation
    
    def get_observation(self):
        """
        Reads the raw vector environment state from the C++ code and
            1) scales it between 0 and 1
            2) checks if the spaceship is wrapping (if that matters)
            3) preprocess the raw vector
        """
        #Read raw vector
        raw_obs = self.get_raw_observation()
        
        #Scale
        scaled_obs = self.scale_observation(raw_obs)
        
        #Check if spaceship is wrapping and penalize
        if self.is_wrapper:
            self.check_wrapping(scaled_obs)
        
        #Preprocessing
        preprocessed_obs = self.preprocess_observation(scaled_obs)
        
        #Make sure that the resulting vector is properly scaled
        assert (preprocessed_obs >= 0).all() and (preprocessed_obs <= 1).all(),\
                    str([raw_obs, scaled_obs, preprocessed_obs])
        return preprocessed_obs
    
    def reset(self):
        """
        Resets the environment. It also loads all the C++ code again so ensure
        a full fresh restart. This is needed because sometimes the game
        goes crazy and the features start being corrputed. The game stays
        corrupted unless a full restart like this is performed. I didn't have time
        to fix this so the workaround consists in calling this function after
        every episode :\
        
        returns:
            observation: first observation of the new game
        """
        self.window_active = False
        self.panel.reset()
        
        self.step_counter = 0
        self.reset_sf()
        
        self.imgs = []
        self.ep_reward = 0
        if self.env_name == 'SF-v0':
            """
            Special handling of mine and shell cords. When they are not on the
            screen they keep their last coordinates until they appear again.
            With these variables we detect when they are gone so we can set their
            coordinates to 0,0 until they reapear to not confuse the agent
            """
            self.last_shell_coords = (0., 0.)
            self.last_mine_coords = (0., 0.)
        
        #Reload game
        self.__init__()
        
        self.configure(self.config)
        
        #Get first observation
        observation = self.get_observation()
#        ship_pos_i = self.get
        return observation # For some reason should show the observation



    def close(self):
        """
        Closes the window of the game and stops execution
        """
        print("Closing!")
        self.stop_drawing()
        sys.exit(0)
        
    def open_window(self):
        cv2.namedWindow(self.env_name)
    
    def define_action_set(self):
        if self.is_no_direction:
            #If no direction is set we reconfigure the key to action mappings
            CT.key_to_action[self.env_name]['Key.down'] = 4
            CT.action_to_sf[self.env_name][4] = CT.key_to_sf['Key.down']
        self._action_set = CT.action_to_sf[self.env_name]
       
        
    def get_prep_feature(self, observation, feature_name):
        index = self.feature_names.index(feature_name)
        return observation[index]
        
    def get_raw_feature(self, observation, feature_name):
        return observation[self.raw_features_name_to_ix[feature_name]]
    
    def define_raw_feature_mappings(self):
        """
        Define dictionary that will be used to get the index of each feature
        in the array of numbers (environment state) that is read from the C++
        """
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
                    'ship_pos_i'            : 0,
                    'ship_pos_j'            : 1,
                    'ship_speed_i'          : 2,
                    'ship_speed_j'          : 3,
                    'ship_headings'         : 4,
                    'missile_pos_i'         : 5,
                    'missile_pos_j'         : 6,
                    'fort_headings'         : 7,
                    'missile_stock'         : 8,
                    'mine_pos_i'            : 9,
                    'mine_pos_j'            : 10,
                    'fortress_lifes'        : 11,
                    'steps_since_last_shot' : 12
                    }
            
    def define_features(self):
        """
        Define the type of features that the agent will receive.
        
        This depends on the SF version of the game and also on game parameters
        such as friction-less, grid-movement, no-direction
        """
        self.define_raw_feature_mappings()
        
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
                    lambda obs: [self.get_raw_feature(obs, 'missile_pos_j')]
#                    lambda obs: [self.get_raw_feature(obs, 'missile_stock')]
                ]
                feature_names += ['ship_pos_i', 'ship_pos_j',
                                  'missile_pos_i', 'missile_pos_j']#,
                                  #'missile_stock']
            
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
        if self.env_name == 'SF-v0':
            #Mines don't wrap even if wrapping is activated
            if self.config.env.mines_activated:
                prep_fs += [
                    lambda obs: [self.get_raw_feature(obs, 'mine_pos_i')],
                    lambda obs: [self.get_raw_feature(obs, 'mine_pos_j')]
                    ]
                feature_names += ['mine_pos_i', 'mine_pos_j']
          
            
            prep_fs += [         
                lambda obs: [self.get_raw_feature(obs, 'fortress_lifes')],
                lambda obs: [self.get_raw_feature(obs, 'steps_since_last_shot')]
                ]
            feature_names += ['fortress_lifes', 'steps_since_last_shot']
          
            

        if self.is_no_direction:
            #Head doesn't control direction, no heading of the spaceship needed
            pass
        else:
            f = lambda obs: aux_decompose_cyclic(self.get_raw_feature(obs, 'ship_headings'))
            prep_fs.append(f)
            feature_names.append("ship_headings_sin")
            feature_names.append("ship_headings_cos")
#            f = lambda obs: [self.get_raw_feature(obs, 'ship_headings')]
#            prep_fs.append(f)
#            feature_names.append('ship_headings')
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
        """
        Configure the space fortress environment.
        
        - Sets some attributes so that they are more handy to use.
        - Loads the C++ functions from the C++ compiled code
        - Set the appropiate action and state spaces
        - Initializes variables
        """
        self.config = cnf
        self.env_name = self.config.env.env_name
       
        
        self.logger = logging.getLogger()
  
        self.scale = 5.3
        
        libpath=self.config.env.library_path

        libname =  '%s_frame_lib_mines%d.so' % \
                                (self.env_name.split('-')[0].lower(),
                                 int(self.config.env.mines_activated))
        
        # Link the environment to the shared libraries
        lib_dir = os.path.join(libpath, libname)
#        print(lib_dir)
        library = ctypes.CDLL(lib_dir)
        #self.logger.info("LOAD "+ lib_dir)
        
        #self.update = library.update_frame
        self.init_game = library.start_drawing
        self.set_key = library.set_key
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

        self.SF_iteration = library.SF_iteration
        self.update_screen = library.update_screen


        self.terminal_state = library.get_terminal_state
        self.score = library.get_score
        self.stop_drawing = library.stop_drawing
        self.pretty_screen = library.get_original_screen

        if self.env_name == 'SF-v0':
            self.did_I_hit_mine = library.did_I_hit_mine
            self.was_I_too_fast = library.was_I_too_fast
            self.did_I_hit_fortress = library.did_I_hit_fortress
            self.did_mine_hit_me = library.did_mine_hit_me
            self.did_fortress_hit_me = library.did_fortress_hit_me
            self.get_vulner_counter = library.get_vulner_counter
            self.get_lifes_remaining = library.get_lifes_remaining
            self.restart_variables = library.restart_variables
            
        sixteen_bit_img_bytes = self.screen_width * self.screen_height * 2
        self.pretty_screen.restype = ctypes.POINTER(ctypes.c_ubyte * sixteen_bit_img_bytes)
        self.score.restype = ctypes.c_float    
        
        # It is possible to specify a seed for random number generation
        self._seed()
        self.define_action_set()

        
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.state_space = gym.spaces.Discrete(self.state_size)

        self.init_game()
        self.episode_dir = os.path.join(self.config.gl.logs_dir,
                                        self.config.model_name, 
                                        'episodes')
        if not self.config.ag == 'human' and not os.path.exists(self.episode_dir):
            os.makedirs(self.episode_dir)
        
        #print("Using features", ', '.join(self.feature_names))
        
       
        font_path = os.path.join(self.config.gl.others_dir, 'Consolas.ttf')
        self.panel = Panel(self.screen_height, font_path)
        
        self.fortress_lifes = self.config.env.fortress_lifes
        self.ship_lifes = self.config.env.ship_lifes
        

    
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
