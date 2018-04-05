
import random
import numpy as np
import sys
import cv2
import gym
import utils
import logging
logger = logging.getLogger(__name__)

def get_gym(env_name):
    import gym_stochastic_mdp            
    env = gym.make(env_name)
    
    return env
class Environment():
    def __init__(self, ag_config, env_config):
        self.env_name = env_config.env_name
        self.gym = get_gym(env_config.env_name)
        
        self.action_size = self.gym.action_size
        self.state_size = self.gym.state_size
        self.action_repeat, self.random_start = \
                env_config.action_repeat, env_config.random_start

        self.display = env_config.display
        
        self._screen = None
        self.reward = 0
        self.terminal = True
        
        #Update configuration
        ag_config.state_size = self.state_size
        ag_config.action_size = self.action_size
        
    def new_game(self, from_random_game=False):
        #if self.lives == 0:
        self._screen = self.gym.reset()
        self.render()
        return self.screen, 0, 0, self.terminal

    def new_random_game(self):
        self.new_game(True)
        for _ in range(random.randint(0, self.random_start - 1)):
            self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def _step(self, action):
        self._screen, self.reward, self.terminal, _ = self.gym.step(action)

    def _random_step(self):
        action = self.gym.action_space.sample()
        self._step(action)

    @property
    def screen(self):
        #return imresize(rgb2gray(self._screen)/255., self.dims)
        #return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_BGR2YCR_CB)/255., self.dims)[:,:,0]
        return self._screen


    @property
    def lives(self):
        #return self.gym.ale.lives()
        return self.gym.lives()

    @property
    def state(self):
        return self.screen, self.reward, self.terminal

    def render(self):
        if self.display:
            self.gym.render()

    def after_act(self, action):        
        self.render()
        
    def act(self, action, is_training=True):
            cumulated_reward = 0
            #start_lives = self.lives
            
            for _ in range(self.action_repeat):
                self._step(action)
                cumulated_reward = cumulated_reward + self.reward
    
                if 0:#is_training and start_lives > self.lives:
                    continue #TODO better understand this
                    cumulated_reward -= 1
                    self.terminal = True
    
                if self.terminal:
                    break
    
            self.reward = cumulated_reward
    
            self.after_act(action)
            
            return self.state
        

                



