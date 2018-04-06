#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

import cv2
import numpy as np


class Environment:
    def __init__(self):
        pass

    def numActions(self):
        # Returns number of actions
        raise NotImplementedError

    def restart(self):
        # Restarts environment
        raise NotImplementedError

    def act(self, action):
        # Performs action and returns reward
        raise NotImplementedError

    def getScreen(self):
        # Gets current game screen
        raise NotImplementedError

    def isTerminal(self):
        # Returns if game is done
        raise NotImplementedError




class GymEnvironment(Environment):
    # For use with Open AI Gym Environment
    def __init__(self, config):
        import gym
        
        self.gym = gym.make(config.env_name)
        if self.gym.spec.id in ["SFS-v0", "SFC-v0", "AIM-v0", "SF-v0"]:
            # Change this to a variable rendering mode
            if config.display_screen == False:
                self.gym.configure(mode="rgb_array", no_direction=config.no_direction)
            else:
                if config.display_screen == True:
                    mode = "human"
                else:
                    mode = config.display_screen

                print("Mode:", mode)

                self.gym.configure(mode=mode,no_direction=config.no_direction,
                                   lib_suffix=config.libsuffix, frame_skip=config.frame_skip)

            
        self.obs = None
        self.terminal = None

        self.screen_width = config.screen_width
        self.screen_height = config.screen_height

    def numActions(self):
        import gym
        assert isinstance(self.gym.action_space, gym.spaces.Discrete)
        return self.gym.action_space.n

    def restart(self):
        self.obs = self.gym.reset()
        self.terminal = False

    def act(self, action):
        self.obs, reward, self.terminal, _ = self.gym.step(action)
        return reward

    def getScreen(self):
        assert self.obs is not None
        self.gym.render()
        if self.gym.spec.id in ["SFS-v0", "SFC-v0", "AIM-v0", "SF-v0"]:
#            print(self.obs.shape)
#            print((self.screen_height, self.screen_width))

#            obs = np.reshape(self.obs, (self.screen_height, self.screen_width))

        #     cv2.imshow("crap", obs)

        #     cv2.waitKey(0)
            # obs = np.reshape(self.obs, (140,140))
            # obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
            return np.reshape(self.obs, (84,84))
        else:
            return cv2.resize(cv2.cvtColor(self.obs, cv2.COLOR_RGB2GRAY), (self.screen_width, self.screen_height))

    def isTerminal(self):
        assert self.terminal is not None
        return self.terminal
