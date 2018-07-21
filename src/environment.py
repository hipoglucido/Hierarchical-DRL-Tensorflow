import gym
from constants import Constants as CT


class Environment():
    """
    Interface between the agent and the gym environments
    Initially taken from:
    https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/environment.py
    but needs readaptation.
    """
    def __init__(self, cnf):
        self.env_name = cnf.env.env_name
        self.gym = self.load_gym()
        self.gym.configure(cnf)
        self.action_size = self.gym.action_space.n
        self.state_size = self.gym.state_space.n
        self.action_repeat = cnf.env.action_repeat

        self.display_prob = cnf.gl.display_prob
        self._screen = None
        self.reward = 0
        self.terminal = True
       
        self.step_counter = 0 #TODO handle training resuming
        
        #Update configuration
        cnf.env.update({"state_size" : self.state_size,
                       "action_size" : self.action_size}, add = True)
        self.gym.reset()
      
    def load_gym(self):
        if self.env_name in CT.GYM_envs:
            gym_env = gym.make(self.env_name).env
            gym_env.configure = lambda x: None
            gym_env.state_space = gym_env.observation_space
            gym_env.state_space.n = gym_env.state_space.shape[0]
            
        elif self.env_name in CT.MDP_envs:
            import gym_stochastic_mdp
            gym_env = gym.make(self.env_name).env
        elif self.env_name in CT.SF_envs:
            import space_fortress
            gym_env = gym.make(self.env_name)#.env
        else:
            assert False, self.env_name + str(CT.GYM_envs)
        #logging.debug("Gym %s built", self.env_name)
        return gym_env
    
    
    def new_game(self, from_random_game=False):
        if self.env_name in CT.SF_envs:
            self.gym.after_episode()
        
        #self.gym = self.load_gym() 
        self._screen = self.gym.reset()
        
        return self.screen, 0., 0., self.terminal


    def _step(self, action):
        self.step_counter += 1
        self._screen, self.reward, self.terminal, self.info = \
                                                    self.gym.step(action)
        self.info['action_repeat'] = self.action_repeat
        self.info['step_counter'] = self.step_counter
        

    def _random_step(self):
        action = self.gym.action_space.sample()
        self._step(action)

    @property
    def screen(self):
        return self._screen

    @property
    def state(self):
        return self.screen, self.reward, self.terminal, self.info

    def render(self):
        self.gym.render()

       
    def act(self, action, info = {}):
        """
        Peforms an action of the gym environment.
        
        params:
            action: int, id of the action to perform. It will be applied
                multiple times on the environment depending on self.action_repeat
                There are a couple of exceptions to this: in the Space Fortress
                gym when using hDQN, even if action_repeat is > 1, when some
                goals are activated action_repeat will be set to 1 because
                otherwise it can be difficult to achieve the goal. Also in SF,
                when the agent is shooting action_repeat will be set to 1 because
                shooting multiple times may be penalized.
            info: dictionary, it contains information like which is the goal
                that is being pursued by the hDQN agent or if the current episode
                needs to be displayed or not.
        """
        cumulated_reward = 0        
        #First we decide if we have to force action_repeat to be 1 or not
        repeat = self.action_repeat
        if self.env_name == 'SF-v0':
            # We force it if the action is to shoot
            if CT.action_to_sf[self.env_name][action] == \
                                            CT.key_to_sf['Key.space']:
                repeat = 1
          
        # Perform the action repeat times            
        for i in range(repeat):            
            self._step(action) # Perform the action
            cumulated_reward = cumulated_reward + self.reward
            should_break = self.extra_checks(i, repeat, info, action)                
            if should_break:
                break
        self.reward = cumulated_reward
        return self.state
    
    def extra_checks(self, i, repeat, info, action):
        
        should_break = False
        ##############################################################
        #  This code is only for the panel of Space Fortress
        #
        if 'goal_name' in info.keys() and self.env_name in CT.SF_envs:
            if self.gym.goal_has_changed:
                self.gym.panel.add(key  = 'actions',
                               item     = '%s:' % info['goal_name'])
                self.gym.panel.add(key  = 'goals',
                                   item = info['goal_name'])
                self.gym.panel.add(key  = 'rewards',
                                   item = ' ')
                self.gym.goal_has_changed = False
        if self.env_name in CT.SF_envs:
            action_name = CT.SF_action_spaces[self.env_name][action]
            self.gym.panel.add(key  = 'actions',
                               item = action_name)
            self.gym.panel.add(key  = 'rewards',
                               item = self.reward)
            import random
            q_value = random.random()
            destroyed = q_value > 0.8
            qinfo = {'destroyed' : destroyed,
                     'q'         : q_value}
            self.gym.qpanel.add(info = qinfo)
        if i != repeat - 1 and (info['display_episode'] or info['watch']) \
                           and self.env_name in CT.SF_envs:
            # In SpaceFortress we render the skipped frames as well
            self.gym.render()
        #
        ##############################################################
        """
        For the goals related with hitting the fortress we check if they have
        been accomplished while frameskipping. It is easier doing this than
        checking it when the MC takes control back.
        """
        lg = ['G_hit_fortress_twice', 'G_hit_fortress_once']
        
        if 'goal_name' in info.keys() and info['goal_name'] in lg:
            if not info['goal'].achieved_inside_frameskip and \
                info['goal'].is_achieved(screen = self.screen,
                                           action = action,
                                           info = self.info):
                
                info['goal'].achieved_inside_frameskip = True
        if self.terminal:
            should_break = True
            
        # Check if is aiming?
            
        return should_break
        
    

                



