# -*- coding: utf-8 -*-
import numpy as np
import time
import re
from configuration import hDQNSettings, DQNSettings
from constants import Constants as CT

class Metrics:
    """
    Class for recording learning metrics. Meant to be written to tensorboard
    periodically
    """
    def __init__(self, config, goals = {}):
        self.config = config
        self.is_hdqn = isinstance(self.config.ag, hDQNSettings) 
        self.is_pmemory = config.ag.pmemory
        self.is_SF = self.config.env.env_name in CT.SF_envs
        self.aux = 0
        lower_bound = -99999
        if self.is_hdqn:        
            self.mc_max_avg_ep_reward = lower_bound
            self.c_max_avg_ep_reward = lower_bound #Not used for now
        else:
            self.max_avg_ep_reward = lower_bound
        self._define_metrics(goals)
        self.restart()
        if self.is_hdqn:
            self.mc_params = config.ag.mc
            self.c_params = config.ag.c
        elif isinstance(self.config.ag, DQNSettings):        
            self.params = config.ag
        else:
            raise ValueError
        self.error_value = -1.

    def _define_metrics(self, goals):
        self.scalar_global_tags = ['elapsed_time', 'games',
                                 'total_episodes', 'debug_states_rfreq_sum',
                                 'debug_no_ep_error', 'progress']
        if self.config.env.env_name == 'SF-v0':
            
            self.scalar_global_tags.append('fortress_hits')
                                           
        if self.is_hdqn:
            #hDQN
            self.mc_scalar_tags = ['mc_step_reward', 'mc_epsilon', 'mc_steps']
            self.c_scalar_tags = ['c_avg_goal_success']#, 'c_steps_by_goal']
            self.scalar_global_tags.append('debug_goals_rfreq_sum')
        else:
            #DQN
            self.ag_scalar_tags = ['step_reward', 'epsilon']
        
        
        self.scalar_dual_tags = ['avg_reward', 'avg_loss', 'avg_q', \
                     'max_ep_reward', 'min_ep_reward', \
                     'avg_ep_reward', 'learning_rate', 'total_reward', \
                     'ep_reward', 'total_loss', 'total_q', 'update_count',
                     'memory_size', 'td_error', 'steps_per_episode']
        if self.is_pmemory:
            self.scalar_dual_tags.append('beta')
        for tag in self.scalar_dual_tags:
            if self.is_hdqn:
                #hDQN
                self.mc_scalar_tags.append("mc_" + tag)
                self.c_scalar_tags.append("c_" + tag)
            else:
                #DQN
                self.ag_scalar_tags.append(tag)
            
        
        self.histogram_global_tags = ['state_visits']
        if self.is_hdqn:
            #hDQN
            self.mc_histogram_tags = ['mc_goals', 'mc_ep_rewards']
            self.c_histogram_tags = ['c_actions', 'c_ep_rewards']   
        else:
            #DQN
            self.ag_histogram_tags = ['actions', 'ep_rewards']
        
        
        
        self.goal_tags = []
        self.state_tags = []
        self.state_names = []
        for k, goal in goals.items():
            name = goal.name
            avg_steps_tag = name + '_avg_steps'
            successes_tag = name + '_successes'
            frequency_tag = name + '_freq'
            relative_frequency_tag = name + '_rfreq'
            success_rate_tag = name + '_success_rate'
            epsilon_tag = name + '_epsilon'
            self.goal_tags += [successes_tag, frequency_tag, success_rate_tag,
                                 relative_frequency_tag, epsilon_tag,
                                 avg_steps_tag]
            assert isinstance(self.config.ag, hDQNSettings)
        
        if not self.is_SF:
            for state_id in range(self.config.env.state_size):
                state_name = "s" + str(state_id)
                self.state_names.append(state_name)
                self.state_tags.append(state_name + "_freq")
                self.state_tags.append(state_name + "_rfreq")
        
        self.scalar_global_tags += self.state_tags
        if self.is_hdqn:
            #hDQN
            self.mc_scalar_tags += self.goal_tags
            self.scalar_tags = self.mc_scalar_tags + self.c_scalar_tags + self.scalar_global_tags
            self.histogram_tags = self.histogram_global_tags + self.mc_histogram_tags + \
                                            self.c_histogram_tags
        else:
            #DQN
            self.scalar_tags = self.ag_scalar_tags + self.scalar_global_tags
            self.histogram_tags = self.histogram_global_tags + self.ag_histogram_tags
        
                                                
    def restart(self):
        
        for s in self.scalar_tags:
#            print(s)
            setattr(self, s, 0.)
        for h in self.histogram_tags:
#            print(h)
            setattr(self, h, [])
        
        try:
            self.restart_timer()
        except AttributeError:
            self.start_timer()
        
    def update_epsilon(self, value, goal_name = None):
        if goal_name is None:
            if self.is_hdqn:
                #Meta-Controller
                self.mc_epsilon = value
            else:
                self.epsilon = value
        else:
            #Controller epsilons are not updated here
            #TODO clean
            pass
        
        
    def store_goal_result(self, goal, achieved):
        name = goal.name
        frequency_tag = name + '_freq'
        setattr(self, frequency_tag, getattr(self, frequency_tag) + 1.)
        if achieved:
            successes_tag = name + '_successes'
            setattr(self, successes_tag, getattr(self, successes_tag) + 1.)
            
    def compute_goal_results(self, goals):
        total_goals_set = 0
        total_achieved_goals = 0
        goal_success_rates = []
        for _, goal in goals.items():
            name = goal.name
            successes = getattr(self, name + '_successes')
            frequency = getattr(self, name + '_freq')
            try:
                success_rate = successes / frequency
            except ZeroDivisionError:                
                success_rate = self.error_value
            setattr(self, name + '_success_rate', success_rate)
            goal_success_rates.append(success_rate)
            total_goals_set += frequency
            total_achieved_goals += successes
            setattr(self, name + "_epsilon", goal.epsilon)
        debug_goals_rfreq_sum = 0
        
        for _, goal in goals.items():
            name = goal.name
            frequency = getattr(self, name + "_freq")
            try:
                rfreq = frequency / total_goals_set
                debug_goals_rfreq_sum += rfreq
            except ZeroDivisionError:
                rfreq = self.error_value
            setattr(self, name + "_rfreq", rfreq)
        setattr(self, 'debug_goals_rfreq_sum', debug_goals_rfreq_sum)   
        try:
            c_avg_goal_success = total_achieved_goals/total_goals_set
        except ZeroDivisionError:
            c_avg_goal_success = self.error_value
        setattr(self,'c_avg_goal_success', c_avg_goal_success)
    
        return c_avg_goal_success
        
    def compute_state_visits(self):
        total_visits = 0
        for state_name in self.state_names:
            visits = getattr(self, state_name + "_freq")
            total_visits += visits
        debug_states_rfreq_sum = 0
        for state_name in self.state_names:
            visits = getattr(self, state_name + "_freq")
            try:
                relative_visits = visits / total_visits
                debug_states_rfreq_sum += relative_visits
            except ZeroDivisionError:
                relative_visits = self.error_value
                
            
            setattr(self, state_name + "_rfreq", relative_visits)
        setattr(self, 'debug_states_rfreq_sum', debug_states_rfreq_sum)
         
    def update_best_score(self):
        if self.is_hdqn:
            self.mc_max_avg_ep_reward = max(self.mc_max_avg_ep_reward,
                                          self.mc_avg_ep_reward)
            self.c_max_avg_ep_reward = max(self.c_max_avg_ep_reward,
                                          self.c_avg_ep_reward)
        else:
            self.max_avg_ep_reward = max(self.max_avg_ep_reward,
                                          self.avg_ep_reward)
        
    def has_improved(self):
        if self.is_hdqn:
            result = self.mc_max_avg_ep_reward * 0.9 <= self.mc_avg_ep_reward
        else:
            result = self.max_avg_ep_reward * 0.9 <= self.avg_ep_reward
        return result
   
    def c_print(self):
        msg = ("\nC: avg_r: {:.4f}, avg_l: {:.6f}, avg_q: {:.3f}, "+\
                            "avg_ep_r: {:.2f}, avg_g: {:.2f}, freq_g5: "+\
                            "{:.2f}, secs: {:.1f}, #g: {}").format(
                                    self.c_avg_reward,
                                    self.c_avg_loss,
                                    self.c_avg_q,
                                    self.c_avg_ep_reward,
                                    self.c_avg_goal_success,
                                    self.g5_freq,
                                    self.elapsed_time,
                                    self.games)
        print(msg)
    def print(self, prefix):
        prefix = prefix + "_" if prefix is not '' else prefix
        msg = "\n" + (prefix.upper() + ": avg_r: {:.4f}, avg_l: {:.6f}, avg_q: {:.3f}, "+\
                            "avg_ep_r: {:.2f}, max_ep_r: {:.2f}, min_ep_r: "+\
                            "{:.2f}, secs: {:.1f}, #g: {}").format(
                                    getattr(self, prefix + "avg_reward"),
                                    getattr(self, prefix + "avg_loss"),
                                    getattr(self, prefix + "avg_q"),
                                    getattr(self, prefix + "avg_ep_reward"),
                                    getattr(self, prefix + "max_ep_reward"),
                                    getattr(self, prefix + "min_ep_reward"),
                                    getattr(self, "elapsed_time"),
                                    getattr(self, "games"))
        print(msg)
        
    def start_timer(self):
        self.t0 = time.time()
        
    def restart_timer(self):
        self.t1 = time.time()
        self.elapsed_time, self.t0 = self.t1 - self.t0, self.t1

    def increment_external_reward(self, external):
        if isinstance(self.config.ag, hDQNSettings):
            #hDQN
            self.mc_total_reward += external        
            self.mc_ep_reward += external
            self.mc_step_reward += external
        else:
            #DQN
            self.total_reward += external        
            self.ep_reward += external
            self.step_reward += external
            
    def increment_rewards(self, internal, external):
        self.c_total_reward += internal
        self.c_ep_reward += internal
        self.increment_external_reward(external)
        
        
        

    def filter_summary(self, summary):
        
        exclude_inside = ['_avg_steps', '_freq']
        exclude_equals = [
                  'mc_ep_reward',
                  'c_ep_reward',
                  'mc_step_reward',
                  'ep_reward',
                  'step_reward',
                  #'avg_reward',
                  #'c_avg_reward',
                  #'mc_avg_reward'
                          ]
        exclude_regex = ['g[0-9]+_successes','g[0-9]+_freq','s[0-9]+_freq','s[0-9]+_rfreq']        
        
        for key in list(summary):
            
            delete = False
            if any([ei in key for ei in exclude_inside]):
                delete = True
            elif any([key == ee for ee in exclude_equals]):
                delete = True
            elif any([re.search(regex, key) for regex in exclude_regex]):
                delete = True
            if delete:
                del summary[key]
           
    def rename_summary(self, summary):
        rename_dict = {'mc_avg_ep_rewward' : 'avg_ep_reward'}
        
        for old, new in rename_dict.items():
#            try:
            summary[old] = new
#            except KeyError:
#                pass
        
    def get_summary(self):
        summary = {}
        tags = self.scalar_tags + self.histogram_tags
        for tag in tags:
            summary[tag] = getattr(self, tag)
          
        return summary
    
    def compute_test(self, prefix, update_count = None, mc_steps = None):
        assert prefix in ['c', 'mc', '']
        prefix = prefix + '_' if prefix != '' else prefix
        update_count = getattr(self, prefix + 'update_count')
        if not update_count > 0:
            #Model hasn't started training
            pass
        
        config = getattr(self, prefix + 'params')
        if prefix == 'mc_':
            test_step = self.mc_steps    
        else:
            test_step = config.test_step
        total_reward = getattr(self, prefix + 'total_reward')
        setattr(self, prefix + 'avg_reward', total_reward / test_step)
        total_loss = getattr(self, prefix + 'total_loss')
        if update_count > 0:
            setattr(self, prefix + 'avg_loss', total_loss / update_count)
            total_q = getattr(self, prefix + 'total_q')
            setattr(self, prefix + 'avg_q', total_q / update_count)
        
        ep_rewards = getattr(self, prefix + 'ep_rewards')
        
        debug_no_ep_error = 0
        try:
            setattr(self, prefix + 'max_ep_reward', np.max(ep_rewards))
            setattr(self, prefix + 'min_ep_reward', np.min(ep_rewards))
            setattr(self, prefix + 'avg_ep_reward', np.mean(ep_rewards))
            self.aux = 1
        except Exception as e:
            print(prefix + ", " + str(e))
            self.aux = 1
            debug_no_ep_error = 1
            for s in ['max', 'min', 'avg']:
                setattr(self, prefix + s +'_ep_reward', self.error_value)
        total_episodes = len(ep_rewards)
        setattr(self, 'total_episodes', total_episodes)
        setattr(self, 'debug_no_ep_error', debug_no_ep_error)
        try:
            steps_per_episode = test_step / total_episodes
        except ZeroDivisionError:
            steps_per_episode = self.error_value
        setattr(self, prefix + 'steps_per_episode', steps_per_episode)
        
    def add_act(self, action, state = None):
        if self.is_hdqn:
            self.c_actions.append(action)
        else:
            self.actions.append(action)
        if self.is_SF:
           return 
        self.state_visits.append(state)
        state_freq_tag = 's' + str(state) + "_freq"
        visits = getattr(self, state_freq_tag)
        setattr(self, state_freq_tag, visits + 1)
        
    def mc_add_update(self, loss, q, td):
        self.mc_total_loss += loss
        self.mc_total_q += q
        self.mc_update_count += 1
        self.mc_td_error += td
    
    def c_add_update(self, loss, q, td):
        self.c_total_loss += loss
        self.c_total_q += q        
        self.c_update_count += 1
        self.c_td_error += td

        
        
    def close_episode(self):
        self.games += 1
        if self.is_hdqn:
            self.mc_ep_rewards.append(self.mc_ep_reward)
            self.c_ep_rewards.append(self.c_ep_reward)
            self.mc_ep_reward, self.c_ep_reward = 0., 0.
        else:
            self.ep_rewards.append(self.ep_reward)
            self.ep_reward = 0.
        
