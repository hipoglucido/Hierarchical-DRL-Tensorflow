import gym
import numpy as np
import cv2
import os
import ctypes
import time
import sys
import math
import logging
from constants import Constants as CT
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class QPanel:
    def __init__(self, height, width, config):
        agent_type = config.ag.agent_type
        self.length = 100
        self.history_keys = ['q', 'win', 'fortress_hit']
        self.history = {}
        for k in self.history_keys:
            self.history[k] = np.zeros(self.length)
        self.height = height
        self.width = width
        if config.env.sparse_rewards:
            self.y_lim = [-2, 10]
        else:
            self.y_lim = [-2, 18]
        if agent_type == 'human':
            self.title = 'Random $\mathcal{R}$'
        elif agent_type == 'dqn':
            self.title = 'Avg $Q$'
        elif agent_type == 'hdqn':
            self.title = 'Avg $Q_{MC}$'
    def add(self, info):
        for key, item in info.items():
            self.history[key][1:] = self.history[key][:-1]
            self.history[key][0] = item
    def get_image(self, info):
        # Draw q values
        dpi= 100#plt.gcf().get_dpi()
        
        
        figure = plt.figure(figsize = (self.width / dpi, self.height / dpi))
        plot = figure.add_subplot(111)
        plot.plot(self.history['q'], color = 'blue', label = self.title)
        
        colors = ['red', 'green']
        keys = ['fortress_hit']
        for color, key in zip(colors, keys):
            for i, boolean in enumerate(self.history[key]):
                if boolean:
                    plot.axvline(x = i, linestyle = '--', color = 'blue',
                                 alpha = .5, linewidth = 1)
                
        plt.ylim(self.y_lim)
        plt.ylabel(self.title)
#        from collections import OrderedDict
#        
#        handles, labels = plt.gca().get_legend_handles_labels()
#        by_label = OrderedDict(zip(labels, handles))
#        plt.legend(by_label.values(), by_label.keys(), loc = 'upper right')
        
        plt.tick_params(axis = 'x', which = 'both', bottom = 0, labelbottom = 0)
        figure.tight_layout()
        
        q_panel = fig2img(figure)
        plt.close()
        return q_panel
class Panel:
    """
    This class only serves for visualization purposes. It appends an image
    to each frame with information about goals, action, rewards etc.
    """
    def __init__(self, height, font_path, agent_type):
        self.length = 17
        self.agent_type = agent_type
        self.history_keys = ['goals', 'actions', 'rewards', 'q']
        
            
        
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
        self.font3 = ImageFont.truetype(font = font_path,
                                             size = 30)
    def reset(self):
        self.history = {}
        for k in self.history_keys:
            self.history[k] = ["..." for _ in range(self.length)]
        
        
    def add(self, key, item):
      
        if not isinstance(item, str) and item % 1 == 0:
            item = int(item)
        item = str(item)
        replacements = [('Key.', ''), ('G_',''), ('space','shoot'),
                        ('up', 'thrust')]
        for old, new in replacements:
            item = item.replace(old, new)
        self.history[key][:-1] = self.history[key][1:]
        self.history[key][-1] = item
    
        
    def get_image(self, info):
        panel = Image.new("RGB", (self.width, self.height), "white")
        color_aux = (150, 25, 25)
        draw = ImageDraw.Draw(panel)
        #Draw goals
        j = int(self.width * .04)
        for n, item in enumerate(self.history['goals']):
            coords = (j, self.history_limit_i + self.span * n)
            if n == self.length - 1:
                color = self.current_color
                item = '> ' + item
            else:
                color = self.old_color
                if n == 0:
                    item = 'GOALS'
                    color = color_aux
            
            draw.text(coords, item, color, font = self.font1)
        #Draw rewards
        if self.agent_type == 'dqn':
            j_rewards = int(self.width * .3)
            j_actions = int(self.width * .6)
        else:
            j_rewards = int(self.width * .52)
            j_actions = int(self.width * .7)
        j_line = int(j_rewards * .9)
        draw.line((j_line, self.history_limit_i, j_line, self.height),
                  fill = self.old_color) 
        
        for n, item in enumerate(self.history['rewards']):
            coords = (j_rewards, self.history_limit_i + self.span * n)
            if n == self.length - 1:
                color = self.current_color
                item = '> ' + item
            else:
                color = self.old_color
                if n == 0:
                    item = 'R'
                    color = color_aux
            draw.text(coords, item, color, font = self.font1)
        
        #Draw actions
        for n, item in enumerate(self.history['actions']):
            #print(n, item)
            coords = (j_actions, self.history_limit_i + self.span * n)
            if n == self.length - 1:
                color = self.current_color
                item = '> ' + item
            else:
                color = self.old_color
                if n == 0:
                    item = 'ACTIONS'
                    color = color_aux
            draw.text(coords, item, color, font = self.font1)
      
        
        j = int(self.width * .3)
        seconds_per_step = 50 / 1000 
        draw.text((10, 5),"%.2fs" % (info['steps'] * seconds_per_step),
                                                  color_aux, font = self.font2)
        draw.text((10, 25),"%d steps" % info['steps'], color_aux, font = self.font1)
        # Draw stuff to debug state of the game
#        draw.text((j, 3), "%.2f, %d, %.2f" % (
#                                          info['mine_present'],
#                                          info['debug2'],
#                                          info['debug3']),
#                                          color_aux, font = self.font1)
        draw.text((j, 3), "Total reward: %.2f" % info['ep_reward'], color_aux, font = self.font1)
        fortress_lifes = info['fortress'] - 1              
        if fortress_lifes < 2:
            msg = '[V]'
            #fortress_lifes = 1
        else:
            msg = ''
        draw.text((j, 20), "Lifes: ship %d fort %d %s" % (info['ship'],
                  fortress_lifes, msg),
                                          color_aux, font = self.font1)

        return panel#q_panel#panel
            
def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf
def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )
class SFEnv(gym.Env):
    """
    Space Fortress Gym
    """
    
  
    def __init__(self):
        self.imgs = []
        self.step_counter = 0
        self.ep_counter = 0
        self.ep_reward = 0
        self.steps_since_mine_appeared = 1e10
        self.steps_since_last_shot = 1e10
        self.steps_since_last_fortress_hit = 1e10
        self.steps_since_last_fortress_hit_aux = 1e10
        self.goal_has_changed = False
        self.print_shit = False
        self.currently_wrapping = False
        self.penalize_wrapping = False
        self.win = False
        
        #Just for metrics
        self.shot_too_fast = False
        
        # For debugging
        self.current_observation = None
        
    
        
    def _seed(self):
        #TODO
        pass
    
    @property
    # Returns the amount of actions
    def n_actions(self):
        return len(self._action_set)


    def after_episode(self):
        self.ep_counter += 1
        if 0 < len(self.imgs):# < 1000:
            self.generate_video()
            
    def get_custom_reward(self, action):
        reward = 0
        cnf = self.config.env
        # Penalize shooting fast at wherever
#        if self.is_shot(action) and \
#                    self.steps_since_last_shot < \
#                                cnf.min_steps_between_shots and \
#                    self.fortress_lifes > 2:
#            reward -= cnf.fast_shooting_penalty
#            
#            self.fortress_lifes = cnf.fortress_lifes            
#            self.shot_too_fast = True
#        else:
#            self.shot_too_fast = False
            
        if self.is_shot(action):
            self.steps_since_last_shot = 0
        else:
            self.steps_since_last_shot += 1

        # Did I hit mine
        if self.did_I_hit_mine() and cnf.mines_activated:
            reward += cnf.hit_mine_reward
        self.shot_too_fast = False
        
        self.steps_since_last_fortress_hit_aux = self.steps_since_last_fortress_hit
        # Did I hit fortress?
        if not self.did_I_hit_fortress() or self.step_counter == 0:
            # Didn't hit the fortress
            self.steps_since_last_fortress_hit += 1
            if self.steps_since_last_fortress_hit > \
                        cnf.min_steps_between_fortress_hits and \
                                        self.fortress_lifes == 1 and \
                                    not cnf.ez:
                # fortress_lifes 1 -> 2
                self.fortress_lifes = 2
        elif not self.did_I_hit_fortress():
            # Didn't hit the fortress
            pass
        # From here on it is asssumed that the fortress was hit
        elif self.fortress_lifes > 1 and self.steps_since_last_fortress_hit <= \
                            cnf.min_steps_between_fortress_hits:
            # Double shoot not allowed
            self.shot_too_fast = True
            reward -= cnf.fast_shooting_penalty
            self.steps_since_last_fortress_hit = 0
            if not cnf.ez:
                # Restart fortress lifes if in hard mode
                self.fortress_lifes = cnf.fortress_lifes
        elif cnf.ez or (not self.mine_present or \
                   self.steps_since_mine_appeared < \
                           cnf.max_steps_after_mine_appear):
            # Playing in EZ mode or mine restrictions don't apply
            if self.fortress_lifes == 1 and \
                    self.steps_since_last_fortress_hit < \
                            cnf.min_steps_between_fortress_hits:
                reward = cnf.final_double_shot_reward
                self.fortress_lifes -= 1
                # WIN!
                self.win = True
            elif self.steps_since_last_fortress_hit > \
                            cnf.min_steps_between_fortress_hits:
                
                self.fortress_lifes -= 1
                if self.fortress_lifes == 0 and cnf.ez:
                    self.win = True
                if self.fortress_lifes > 1 or cnf.ez:
                    reward += cnf.hit_fortress_reward
                
                
            else:
                # You shoot too fast when it was not allowed
                pass#self.fortress_lifes += 1           
            self.steps_since_last_fortress_hit = 0
        else: 
            self.steps_since_last_fortress_hit = 0
            pass
            
    
            
        # Bad
        if self.did_mine_hit_me() and cnf.mines_activated:
            reward -= cnf.hit_by_mine_penalty
            self.ship_lifes -= 1
        if self.did_fortress_hit_me():
            reward -= cnf.hit_by_fortress_penalty
            self.ship_lifes -= 1
        
        
        # Penalize wrapping
        if self.penalize_wrapping:
            reward -= cnf.wrapping_penalty
        
        if reward == 0:
            # Time penalty
            reward -= cnf.time_penalty
      
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
        elif self.ship_lifes == 0:
            is_terminal = True
        elif self.shot_too_fast:
            pass#is_terminal = True
        return is_terminal
    
    def step(self, action):
        """
        Performs one action on the environment. Always atomic actions 
        (granularity = 1). This method is agnostic to the
        action_repeat / frameskip that is being used by the agent.
        """
        
        #Call the C++ function
        self.perform_action(action)   
        
        aux = int(self.steps_since_last_shot) # Needed for the double_shoot goal
        reward = self.get_custom_reward(action)

        done = self.is_terminal()
        destroyed = int(self.fortress_lifes == 0)
        # This info is sent back to the environment.py and agent to compute
        # stuff related with metrics, achievement of goals etc.
        info = {
            'fortress_hit'                  : self.did_I_hit_fortress(),
            'mine_hit'                      : self.did_I_hit_mine(),
            'hit_by_fortress'               : self.did_fortress_hit_me(),
            'hit_by_mine'                   : self.did_mine_hit_me(),
            'win'                           : int(self.win),
            'destroyed'                     : destroyed,
            'steps'                         : self.step_counter, 
            'wrap_penalization'             : int(self.penalize_wrapping),
            'shot_too_fast_penalization'    : int(self.shot_too_fast),
            'steps_since_last_shot'         : aux,
            'min_steps_between_shots'       : int(self.config.env.min_steps_between_shots),
            'steps_since_last_fortress_hit' : self.steps_since_last_fortress_hit,
            'steps_since_last_fortress_hit_aux' : self.steps_since_last_fortress_hit_aux
        }
                
        self.restart_variables()
        self.step_counter += 1
        self.ep_reward += reward
        observation = self.get_observation() 
        return observation, reward, done, info
    
    def scale_observation(self, raw_obs):
        #print(raw_obs, len(raw_obs))
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
            #Features coming from the C game
            raw_obs[0]  /= self.screen_height         # Ship_Y_Pos
            raw_obs[1]  /= self.screen_width          # Ship_X_Pos
            raw_obs[2]   = (raw_obs[2] + 5.) / 10.    # Ship_Y_Speed
            raw_obs[3]   = (raw_obs[3] + 5.) / 10.    # Ship_X_Speed 
            raw_obs[4]  /= 360                        # Ship_Headings
            raw_obs[5]  /= self.screen_height         # Shell_Y_Pos
            raw_obs[6]  /= self.screen_width          # Shell_X_Pos
            raw_obs[7]  /= 360                        # fort_Headings
            raw_obs[8]  /= 100                        # Missile_Stock
            raw_obs[9]  /= self.screen_height         # Mine_Y_Pos
            raw_obs[10] /= self.screen_width          # Mine_X_Pos
            #Features coming from this environment
            raw_obs[12] = (raw_obs[12] + 1) / \
                 (self.config.env.fortress_lifes + 1) # Fortress lives
            raw_obs[13] = np.tanh(raw_obs[13] * .1)   # Steps_since_last_shot
            raw_obs[14] = np.tanh(raw_obs[14] * .01)  # Steps since mine appeared

            # If shells are away from the screen, put them as 0, 0
            if self.last_shell_coords == (raw_obs[5], raw_obs[6]):
                raw_obs[5], raw_obs[6] = 0., 0.
            else:
                self.last_shell_coords = (raw_obs[5], raw_obs[6])
            # Same with mines
            if self.last_mine_coords == (raw_obs[9], raw_obs[10]) or \
                not self.config.env.mines_activated:
                raw_obs[9], raw_obs[10] = 0., 0.
            else:
                self.last_mine_coords = (raw_obs[9], raw_obs[10])                      
        raw_obs = np.clip(raw_obs, 0, 1)
        return raw_obs
    
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
                    'fortress_lifes'        : 12,
                    'steps_since_last_shot' : 13,
                    'steps_since_mine_appeared' : 14
                    }
              
    
            
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
        return time.strftime("%dd%Hh%Mm%Ss", time.gmtime())

    
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
        
        # This information will be send to the panel (only for display purpose)
        info = {'steps'        : self.step_counter,
                'ship'         : self.ship_lifes,
                'fortress'     : self.fortress_lifes,
                'mine_present' : self.mine_present,
                'ep_reward'    : self.ep_reward,
                'debug1'       : self.steps_since_mine_appeared,
                'debug2'       : self.get_prep_feature(self.current_observation, 
                                                            'steps_since_last_shot')                          
                                                    ,
                'debug3'       : self.get_prep_feature(self.current_observation, 
                                                            'fortress_lifes')                          
                                                    }
        
        panel_img = self.panel.get_image(info)
        qpanel_img = self.qpanel.get_image(info)
        width = self.screen_width + self.panel.width
        height = self.screen_height
        full_image = Image.new('RGB', (width, height + 100))
        full_image.paste(env_img, (0, 0))
        full_image.paste(panel_img, (self.screen_width, 0))
        full_image.paste(qpanel_img, (0, self.screen_height))
        
        if not self.config.gl.watch:
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
        video_name = "%dsteps_%s_R%.2f_win%d.mp4" % \
            (self.step_counter, self.current_time, self.ep_reward, int(self.win))
        video_path = os.path.join(self.episode_dir, video_name)
        (original_width, original_heigth) = self.imgs[0].size
        # PIL image >>> np.array
        self.imgs = [np.array(img) for img in self.imgs]
        #Adds a white frame at the end (useful if .gif is produced)
        last_frame = self.imgs[-1].copy()
        self.imgs = self.imgs + 10 * [last_frame]
        #Make video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, 20.0, (original_width, original_heigth))
        #imageio.mimsave(video_path, self.imgs, duration = .00001)
        for img in self.imgs:
            video.write(img)
        video.release()
    def check_mine_present(self, obs):
        """
        Checks if there is a mine present
        """
        i = self.get_raw_feature(obs, 'mine_pos_i')
        j = self.get_raw_feature(obs, 'mine_pos_j')
        if i == 0 and j == 0:
            # There is no mine in the screen
            self.mine_present = False            
        else:
            # There is a mine in the screen
            if not self.mine_present:
                # The mine has just appeared
                self.steps_since_mine_appeared = 0
            else:
                # The mine was already present
                pass
            self.mine_present = True
        
        
        if not self.mine_present:
            self.steps_since_mine_appeared = 1e10
        else:
            self.steps_since_mine_appeared += 1
            self.mine_present = True
            
        
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
   
    def get_raw_observation(self):  
        # From C++ game
        game_obs = np.ctypeslib.as_array(self.get_symbols().contents)
        
        # Extra features
        extra_obs = np.array([self.fortress_lifes,
                              self.steps_since_last_shot,
                              self.steps_since_mine_appeared],
                              dtype = np.float)
        
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
        self.check_mine_present(scaled_obs)
        #Preprocessing
        preprocessed_obs = self.preprocess_observation(scaled_obs)
        
        #Make sure that the resulting vector is properly scaled
        assert (preprocessed_obs >= 0).all() and (preprocessed_obs <= 1).all(),\
                    str([raw_obs, scaled_obs, preprocessed_obs])
                    
        # For debugging
        self.current_observation = preprocessed_obs
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
        self.stop_drawing()
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
        """
        Sometimes the game crashes and nothing makes sense anymore. As for now,
        it is hard to know why and detect how. It is fixed by restarting the
        game        
        """
        q_history = self.qpanel.history.copy()
        self.__init__()
        
        self.configure(self.config)
        self.qpanel.history = q_history
        #Get first observation
        observation = self.get_observation()
        return observation


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
        """
        Gets preprocessed feature
        """
        index = self.feature_names.index(feature_name)
        return observation[index]
        
    def get_raw_feature(self, observation, feature_name):
        """
        Gets raw feature
        """
        return observation[self.raw_features_name_to_ix[feature_name]]
   
    def define_features(self):
        """
        Define the type of features that the agent will receive.
        
        This depends on the SF version of the game and also on game parameters
        such as friction-less, grid-movement, no-direction
        """
        self.define_raw_feature_mappings()
        
        """
        These two list are aligned. For each position, there is a feature name
        (string) and a preprocessing function (lambda x) that extracts it.
        NB: one feature name, if cyclic, will be decomposed in two different
        features (_sin and _cos) by its preprocessinf function, so the length
        of these lists is not necessarily equals to the state space that the
        agent sees.
        """
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
                                  'missile_pos_i', 'missile_pos_j']
            
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
                    lambda obs: [self.get_raw_feature(obs, 'missile_pos_j')]
#                    lambda obs: [self.get_raw_feature(obs, 'missile_stock')]
                ]
                feature_names += ['missile_pos_i', 'missile_pos_j']#,
#                                  'missile_stock']
        if self.is_frictionless and self.env_name != 'AIM-v0':
                # FRICTIONLESS
                prep_fs += [
                    lambda obs: [self.get_raw_feature(obs, 'ship_speed_i')],
                    lambda obs: [self.get_raw_feature(obs, 'ship_speed_j')]
                    ]
                feature_names += ['ship_speed_i', 'ship_speed_j']          
        if self.env_name == 'SF-v0':
            #Mines don't wrap even if wrapping is activated
            if self.config.env.mines_activated or 1:
                # Include even if not activated for potential transfer learning
                prep_fs += [
                    lambda obs: [self.get_raw_feature(obs, 'mine_pos_i')],
                    lambda obs: [self.get_raw_feature(obs, 'mine_pos_j')]]

                feature_names += ['mine_pos_i', 'mine_pos_j']
                if not self.config.env.ez:
                    # Not relevant in ez mode
                    prep_fs += [
                    lambda obs: [self.get_raw_feature(obs, 'steps_since_mine_appeared')]
                        ]
                    feature_names += ['steps_since_mine_appeared']
            
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
                                        self.config.ag.experiment_name,
                                        self.config.model_name, 
                                        'episodes')
        if not self.config.ag == 'human' and not os.path.exists(self.episode_dir):
            os.makedirs(self.episode_dir)
        
        #print("Using features", ', '.join(self.feature_names))
        font_path = os.path.join(self.config.gl.others_dir, 'Consolas.ttf')
        self.panel = Panel(self.screen_height, font_path, self.config.ag.agent_type)
        self.qpanel = QPanel(height = 100,
                             width = self.screen_width + self.panel.width,
                             config = self.config)
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
