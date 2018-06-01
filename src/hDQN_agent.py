from __future__ import print_function
import os
import time
import random
import numpy as np
from tqdm import tqdm
from functools import reduce
import tensorflow as tf
import sys

from base import Agent, Epsilon
from history import History
from replay_memory import ReplayMemory, PriorityExperienceReplay, OldReplayMemory
from utils import linear, huber_loss, weighted_huber_loss
from utils import get_time, save_pkl, load_pkl, pp
from goals import MDPGoal, generate_SF_goals
from metrics import Metrics
from configuration import Constants as CT

        
class HDQNAgent(Agent):
    def __init__(self, config, environment, sess):
        super().__init__(config)
        self.sess = sess
        self.weight_dir = 'weights'

        self.config = config
        self.ag = self.config.ag
        self.c_ag = self.ag.c
        self.mc_ag = self.ag.mc
        self.gl = self.config.gl
        #print(self.mc)
        self.environment = environment
        self.goals = self.define_goals()
        
        self.mc_ag.update({"q_output_length" : self.ag.goal_size}, add = True)
        self.c_ag.update({"q_output_length" : self.environment.action_size}, add = True)
        
      
        self.mc_history = History(length_ = self.mc_ag.history_length,
                                  size    = self.environment.state_size)
        
        self.c_history = History(length_ = self.c_ag.history_length,
                                 size    = self.environment.state_size)
        memory_type = PriorityExperienceReplay if self.ag.pmemory else ReplayMemory
        self.mc_memory = memory_type(config       = self.mc_ag,
                                      screen_size  = self.environment.state_size)
        self.c_memory = memory_type(config        = self.c_ag,
                                      screen_size  = self.environment.state_size + \
                                                          self.ag.goal_size)
            
        self.m = Metrics(self.config, self.goals)
        
        
        
        #self.config.print()
        self.build_hdqn()
        self.write_configuration()
        #TODO turn config inmutable
    
    
    def get_goal(self, n):
        return self.goals[n]
        
    def define_goals(self):
        
        if self.environment.env_name in CT.MDP_envs:
            self.ag.goal_size = self.environment.state_size
            goals = {}
            for n in range(self.ag.goal_size):
                goal_name = "g" + str(n)
                goal = MDPGoal(n, goal_name, self.c)            
                goal.setup_one_hot(self.ag.goal_size)
                goals[goal.n] = goal
        elif self.environment.env_name in CT.SF_envs:
            #Space Fortress
            goal_names = CT.goal_groups[self.environment.env_name][self.ag.goal_group]
            goals = generate_SF_goals(
                    environment = self.environment,
                    goal_names  = goal_names)
            self.ag.goal_size = len(goals)
            
        else:
            raise ValueError("No prior goals for " + self.environment.env_name)
        self.goal_probs = np.array([1 / len(goals) for _ in goals])
        return goals
    
    def set_next_goal(self, obs):
#        step = self.c_step #TODO think about this
#        if not self.c_learnt or 1:
#            a = list(self.goals.keys())
#            p = self.goal_probs
#            
#            n_goal = random.randrange(self.ag.goal_size)#np.random.choice(a = a, p = p)
#        else:

        if self.is_ready_to_learn(prefix = 'mc'):
            ep = self.mc_epsilon.steps_value(self.c_step)
#            print("____________")
#            print("step",self.c_step)
#            print("learn_start",self.mc_epsilon.learn_start)
#            print("start", self.mc_epsilon.start)
#            print("end",self.mc_epsilon.end)
#            print("end_t",self.mc_epsilon.end_t)
#            print("ep",ep)
        else:
            ep = 1
        
        self.m.update_epsilon(value = ep)
        if random.random() < ep and not self.is_playing():
            
            n_goal = random.randrange(self.ag.goal_size)
        else:
            
            n_goal = self.mc_q_action.eval({self.mc_s_t: [[obs]]})[0]
        self.mc_old_obs = obs
#        n_goal = 5
        self.m.mc_goals.append(n_goal)
        goal = self.get_goal(n_goal)
        goal.set_counter += 1
        self.current_goal = goal
        self.environment.gym.goal_has_changed = True
       
            
        
    def predict_next_action(self, obs):
        
        
        
        #s_t should have goals and screens concatenated
        ep = self.current_goal.epsilon

        if random.random() < ep and not self.is_playing():
            action = random.randrange(self.environment.action_size)
            
        else:
            #screens = self.c_history.get()
            
            action = self.c_q_action.eval(
                            {self.c_s_t: [[obs]],
                             self.c_g_t: [self.current_goal.one_hot]})[0]
    
    
#        action = int(self.aux(self.c_history.get()[-1]) < self.current_goal.n)
#        print('**',self.aux(self.c_history.get()[-1]), self.current_goal.n, action)
        self.m.c_actions.append(action)
        return action
    def mc_observe(self, goal_n, ext_reward, new_obs, terminal):
        if self.is_playing():
            return
            
#        old_state = self.mc_history.get()
#        self.mc_history.add(screen)
#        print("____________mcobs________________")
#        f=self.config.env.factor
#        print("s0:\n",self.mc_old_obs.reshape(f,f))
#        print("g:", self.current_goal.n)
#        print("s1:\n", new_obs.reshape(f,f))
#        print("EXtR:", ext_reward,", terminal", int(terminal))
        self.mc_memory.add(self.mc_old_obs, goal_n, ext_reward, new_obs, terminal)
        
        
        
    
    def c_observe(self, old_obs, action, int_reward, new_obs, terminal):
        if self.is_playing():
            return
        #print("C ", int_reward, "while", self.aux(screen), "a:", action)
        # NB! screen is post-state, after action and reward
#        old_screen = self.c_history.get()[0]
#        print(old_screen)
#        print(self.current_goal.one_hot)
        old_state = np.hstack([self.current_goal.one_hot, old_obs])
#        self.c_history.add(screen)
        next_state = np.hstack([self.current_goal.one_hot, new_obs])
#        if terminal==0 and int_reward==1:
#        print("_______***___________")
#        f=self.config.env.factor
#        print("s_t-1\n", old_obs.reshape(f,f))
#        print("a",action)
#        print("s_t\n",new_obs.reshape(f,f))
#        print("g_t:",self.current_goal.n)#one_hot.reshape((3,3)))
#        print("R: %.2f, t=%s" % (int_reward, str(terminal)))
        
        self.c_memory.add(old_state, action, int_reward, next_state, terminal)
        #Update C
#        if self.is_ready_to_learn(prefix = 'c'):
#            if self.c_step % self.c_ag.train_frequency == 0:
#                self.c_q_learning_mini_batch()
#
#            if self.c_step % self.c_ag.target_q_update_step == self.c_ag.target_q_update_step - 1:
#                self.c_update_target_q_network()
        self.learn_if_ready(prefix = 'c')
        #Update MC   
        self.learn_if_ready(prefix = 'mc')
#        if self.is_ready_to_learn(prefix = 'mc'):
#            
#            if self.mc_step % self.mc_ag.train_frequency == 0:
#                self.mc_q_learning_mini_batch()
#
#            if self.mc_step % self.mc_ag.target_q_update_step ==\
#                        self.mc_ag.target_q_update_step - 1:
#                self.mc_update_target_q_network()
#                
#            if not self.mc_ready:
#                self.mc_epsilon.learn_start = self.c_step
#                self.mc_ready = True

        
    def mc_q_learning_mini_batch(self):      
#        s_t, goal, ext_reward, s_t_plus_1, terminal = self.mc_memory.sample()
        (s_t, goal, ext_reward, s_t_plus_1, terminal), idx_list, p_list, \
                                        sum_p, count = self.mc_memory.sample()
        if self.m.aux:
            pass#print(s_t, goal, ext_reward, s_t_plus_1, terminal)
#        if ext_reward[0] > 0:
#            print("_______---___________")
#            f=self.config.env.factor
#            print("s_t-1\n", s_t[0].reshape(f,f))
#            print("g",goal[0])
#            print("s_t\n",s_t_plus_1[0].reshape(f,f))
#            print("R: %.2f, t=%s" % (ext_reward[0], str(terminal[0])))
        target_q_t = self.generate_target_q_t(prefix       = 'mc',
                                              reward       = ext_reward,
                                              s_t_plus_1   = s_t_plus_1,
                                              terminal     = terminal)
        feed_dict = {
            self.mc_target_q_t: target_q_t,
            self.mc_action: goal,
            self.mc_s_t: s_t,
            self.mc_learning_rate_step: self.mc_step,
        }
        
        if self.ag.pmemory:
            beta = (1 - self.mc_epsilon.steps_value(self.c_step)) + self.mc_epsilon.end
            self.m.mc_beta = beta
            loss_weight = (np.array(p_list)*count/sum_p)**(-beta)
            feed_dict[self.mc_loss_weight] = loss_weight
            
            pass
        
        
        _, q_t, mc_td_error, loss = self.sess.run([self.mc_optim,
                                                             self.mc_q,
                                                             self.mc_td_error,
                                                             self.mc_loss],
                                                             #self.mc_q_summary],
                                                            feed_dict)
        
#        self.writer.add_summary(summary_str, self.mc_step)
        
#        if loss > 1000:
#            print("MC",loss)
#            for fn, v in zip(self.environment.gym.feature_names, s_t[0][0]):
#                print(fn,v)
#            assert 0
        self.m.mc_add_update(loss, q_t.mean(), mc_td_error.mean())
        

    def c_q_learning_mini_batch(self):
        
        
        (s_t, action, int_reward, s_t_plus_1, terminal), idx_list, p_list, \
                                        sum_p, count = self.c_memory.sample()
        
        #TODO: optimize goals in memory
        g_t = np.vstack([g[0] for g in s_t[:, :, :self.ag.goal_size]]) 
        s_t = s_t[:, :, self.ag.goal_size:]
        
        
        g_t_plus_1 = np.vstack([g[0] for g in s_t_plus_1[:, :, :self.ag.goal_size]])
        s_t_plus_1 = s_t_plus_1[:, :, self.ag.goal_size:]
        
#        if int_reward[0] > 0 or 1:
#            print("_______***___________")
#            f=self.config.env.factor
#            print("s_t-1\n", s_t[0].reshape(f,f))
#            print("a",action[0])
#            print("s_t\n",s_t_plus_1[0].reshape(f,f))
#            print("g_t-1\n",g_t[0].reshape(f,f))#one_hot.reshape((3,3)))
#            print("R: %.2f, t=%s" % (int_reward[0], str(terminal[0])))
#        
# 
        
#        q_t_plus_1 = self.c_target_q.eval({
#                                    self.c_target_s_t: s_t_plus_1,
#                                    self.c_target_g_t: g_t_plus_1,
#                                     })
#        
#        terminal = np.array(terminal) + 0. #Boolean to float
#    
#        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
#        target_q_t = (1. - terminal) * self.c_ag.discount * max_q_t_plus_1 + int_reward
#
        #print(s_t_plus_1,terminal,g_t_plus_1)
        target_q_t = self.generate_target_q_t(prefix       = 'c',
                                              reward       = int_reward,
                                              s_t_plus_1   = s_t_plus_1,
                                              terminal     = terminal,
                                              g_t_plus_1   = g_t_plus_1)
        feed_dict = {
            self.c_target_q_t: target_q_t,
            self.c_action: action,
            self.c_s_t: s_t,
            self.c_g_t: g_t,
            self.c_learning_rate_step: self.c_step,
        }
        if self.ag.pmemory:
            beta = (1 - self.mc_epsilon.steps_value(self.c_step)) + self.mc_epsilon.end
            self.m.c_beta = beta
            loss_weight = (np.array(p_list)*count/sum_p)**(-beta)
            feed_dict[self.c_loss_weight] = loss_weight
            pass
            
        _, q_t, c_td_error, loss, loss_aux = self.sess.run([self.c_optim,
                                                               self.c_q,
                                                               self.c_td_error,
                                                               self.c_loss,
                                                               self.c_loss_aux],
                                                               #self.c_q_summary],
                                                            feed_dict)
#        self.writer.add_summary(summary_str, self.c_step)
#        for s,l in zip(s_t, loss_aux):
#            if l > 5:
#                print(l)
#                for fn, v in zip(self.environment.gym.feature_names, s[0]):
#                    print(fn,v)  
#        if loss > 1000 and self.c_step > 10000:
#            print("C",loss)
#            for fn, v in zip(self.environment.gym.feature_names, s_t[0][0]):
#                print(fn,v)
#            assert 0
#        else:
#            print('C', loss)
        self.m.c_add_update(loss, q_t.mean(), c_td_error.mean())


    def play(self):
        self.train()
    
    def train(self):
        #Auxiliary flags (for monitoring)
        self.mc_flag_start_training, self.c_flag_start_training = False, False
        self.c_learnt = False  #Indicates if C already knows how to achieve goals
        
        #Initial step (only useful when resuming training)
        self.mc_start_step = self.mc_step_op.eval()
        self.c_start_step = self.c_step_op.eval()
        
        #Total steps for the session
        total_steps = self.ag.max_step + self.c_start_step
        
        #Set up epsilon ofr MC (e-greedy, linear decay)
        self.mc_epsilon = Epsilon()
        self.mc_epsilon.setup(self.mc_ag, total_steps)
        #Set up epsilon for C (one per goal)
#        for key, goal in self.goals.items():
#            goal.setup_epsilon(self.c_ag, self.c_start_step) 
#        
        old_obs = self.new_episode()
        
        self.m.start_timer()
        # Initial goal
        self.mc_step = self.mc_start_step
        self.c_step = self.c_start_step
        self.set_next_goal(old_obs)
        
        
        if self.m.is_SF and self.gl.paralel == 0:   
            iterator = tqdm(range(self.c_start_step, total_steps),
                                                  ncols=70, initial=self.c_start_step)
        else:
            iterator = range(self.c_start_step, total_steps)
#        print("\nFilling c_memory (%d) and mc_memory (%d) with random experiences..." % \
#                      (self.c_ag.memory_size, self.mc_ag.memory_size))
#        
        for self.c_step in iterator:
#            if self.c_memory.is_full() and self.c_step == self.c_ag.memory_size:
#                print("\nController memory full, MC is at %.2f..." % \
#                              (self.mc_memory.count / self.mc_ag.memory_size))
#                time.sleep(.5)
            
            # Controller acts
            action = self.predict_next_action(old_obs)
            info = {'goal_name'       : self.current_goal.name,
                    'is_SF'           : self.m.is_SF,
                    'display_episode' : self.display_episode}
            
            new_obs, ext_reward, terminal, info = self.environment.act(
                                        action = action,
                                        info   = info)
            self.process_info(info)            
            self.m.add_act(action, self.environment.gym.one_hot_inverse(new_obs))
            
            
        
            goal_achieved = self.current_goal.is_achieved(new_obs, action)
            int_reward = 1. if goal_achieved else 0.
            int_reward -= self.c_ag.intrinsic_time_penalty
            self.c_observe(old_obs, action, int_reward, new_obs, terminal or goal_achieved)
            
            if self.display_episode:
                self.console_print(old_obs, action, ext_reward, int_reward)

            
            self.m.increment_rewards(int_reward, ext_reward)
            ######################
            if terminal or goal_achieved:
                
                self.current_goal.finished(self.m, goal_achieved)
                # Meta-controller learns                
                self.mc_observe(goal_n     = self.current_goal.n,
                                ext_reward = self.m.mc_step_reward,
                                new_obs    = new_obs,
                                terminal   = terminal)
                reward = self.m.mc_step_reward
                self.m.mc_step_reward = 0    
                
                if terminal:
                    if self.display_episode:
                        self.console_print_terminal(reward, new_obs)
                    self.m.close_episode()
                    old_obs = self.new_episode()
                    #self.rebuild_environment()
                else:
                    old_obs = new_obs.copy()
                    
#                print("This", self.m.mc_step_reward)
                    
                self.mc_step += 1
                
                # Meta-controller sets goal
                self.set_next_goal(old_obs)
                self.m.mc_goals.append(self.current_goal.n)
            if not terminal:
                old_obs = new_obs.copy()

#            if not self.is_ready_to_learn(prefix = 'c'):#c_step < self.c_ag.learn_start:
#                continue
#            if not self.is_testing_time(self.c_step, self.c_ag.test_step):
#                continue
            if not self.is_testing_time(prefix = 'c'):
                continue
            self.m.progress = self.c_step / total_steps
            self.m.compute_test('c', self.m.c_update_count)
            self.m.compute_test('mc', self.m.mc_update_count, self.mc_step)
            goal_success_rate = self.m.compute_goal_results(self.goals)
            self.c_learnt = goal_success_rate > self.c_ag.learnt_threshold
#            probs = 1. - goal_success_rates.clip(.01, .99)
#            probs = (probs - probs.min()) / (probs.max() - probs.min())
#            probs /= probs.sum()
#            self.goal_probs = probs
            
            
            self.m.compute_state_visits()
            
#            self.m.print('mc')
#            self.m.c_print()
            
            #if self.c_step % self.ag.save_step == 0:
            if self.m.has_improved():
                self.c_step_assign_op.eval(
                        {self.c_step_input: self.c_step + 1})
                self.mc_step_assign_op.eval(
                        {self.mc_step_input: self.mc_step + 1})
                self.delete_last_checkpoints()
                self.save_model(self.c_step + 1)
                self.save_model(self.mc_step + 1)
                self.m.update_best_score()
                

#            if self.c_step > 50:
            self.send_some_metrics(prefix = 'mc')
            self.send_some_metrics(prefix = 'c')
            summary = self.m.get_summary()
            self.m.filter_summary(summary)
            self.m.rename_summary(summary)
            self.inject_summary(summary, self.c_step)
            self.write_output()

            self.m.restart()
            
 

        
        
    def build_meta_controller(self):
        self.mc_w = {}
        self.mc_target_w = {}
        # training meta-controller network
        with tf.variable_scope('mc_prediction'):
            
            self.mc_s_t = tf.placeholder("float",
                        [None, self.mc_history.length, self.environment.state_size],
                        name='mc_s_t')
#            print(self.mc_s_t)
            shape = self.mc_s_t.get_shape().as_list()
            self.mc_s_t_flat = tf.reshape(self.mc_s_t, [-1, reduce(
                                            lambda x, y: x * y, shape[1:])])            
            
            last_layer = self.mc_s_t_flat
            last_layer, histograms = self.add_dense_layers(
                                        architecture = self.mc_ag.architecture,
                                        input_layer = last_layer,
                                        parameters  = self.mc_w,
                                        name_aux = '')
            if self.ag.dueling:
                self.mc_q = self.add_dueling(prefix = 'mc', input_layer = last_layer)
            else:
                self.mc_q, self.mc_w['q_w'], self.mc_w['q_b'] = linear(
                                                          last_layer,
                                                          self.ag.goal_size,
                                                          name='mc_q')
            self.mc_q_action= tf.argmax(self.mc_q, axis=1)
            
            q_summary = histograms
            avg_q = tf.reduce_mean(self.mc_q, 0)
            
#            print(self.mc_ag.q_output_length, avg_q)
            for idx in range(self.mc_ag.q_output_length):
#                print(idx)
#                print(avg_q[idx])
                q_summary.append(tf.summary.histogram('mc_q/%s' % idx, avg_q[idx]))
            self.mc_q_summary = tf.summary.merge(q_summary, 'mc_q_summary')

        # target network
        self.create_target(config = self.mc_ag)

        #Meta Controller optimizer
        self.build_optimizer(prefix = 'mc')

        
        
    def build_controller(self):
        self.c_w = {}
        self.c_target_w = {}
    
        with tf.variable_scope('c_prediction'):
            #input_size = self.environment.state_size + self.ag.goal_size
            
            self.c_s_t = tf.placeholder("float",
                                [None, self.c_history.length,
                                                 self.environment.state_size],
                                name = 'c_s_t')
            shape = self.c_s_t.get_shape().as_list()
            self.c_s_t_flat = tf.reshape(self.c_s_t, [-1, reduce(
                    lambda x, y: x * y, shape[1:])])
            self.c_g_t = tf.placeholder("float",
                               [None, self.ag.goal_size],
                               name = 'c_g_t')
            self.c_gs_t = tf.concat([self.c_g_t, self.c_s_t_flat],
                           axis = 1,
                           name = 'c_gs_concat')
#            print(self.c_g_t)
#            print(self.c_s_t_flat)
            last_layer = self.c_gs_t
#            print(last_layer)
            last_layer, histograms = self.add_dense_layers(
                                            architecture = self.c_ag.architecture,
                                               input_layer = last_layer,
                                               parameters = self.c_w,
                                               name_aux= '')
            if self.ag.dueling:
                self.c_q = self.add_dueling(prefix = 'c', input_layer = last_layer)
            else:
                self.c_q, self.c_w['q_w'], self.c_w['q_b'] = linear(last_layer,
                                                  self.environment.action_size,
                                                  name='c_q')
#            print(self.c_q)
            self.c_q_action= tf.argmax(self.c_q, axis=1)
            
            q_summary = histograms
            avg_q = tf.reduce_mean(self.c_q, 0)
            

            for idx in range(self.c_ag.q_output_length):
                q_summary.append(tf.summary.histogram('c_q/%s' % idx, avg_q[idx]))
            self.c_q_summary = tf.summary.merge(q_summary, 'c_q_summary')

        # target network
        self.create_target(self.c_ag)
        
        
        #Controller optimizer
        self.build_optimizer(prefix = 'c')

    def build_hdqn(self):

        
        with tf.variable_scope('c_step'):
            self.c_step_op = tf.Variable(0, trainable=False, name='c_step')
            self.c_step_input = tf.placeholder('int32', None,
                                                  name='c_step_input')
            self.c_step_assign_op = \
                        self.c_step_op.assign(self.c_step_input)
            
        with tf.variable_scope('mc_step'):
            self.mc_step_op = tf.Variable(0, trainable=False,
                                            name='mc_step')
            self.mc_step_input = tf.placeholder('int32', None,
                                           name='mc_step_input')
            self.mc_step_assign_op = \
                        self.mc_step_op.assign(self.mc_step_input)
                
        
#        print("Building meta-controller")
        self.build_meta_controller()
#        print("Building controller")
        
        self.build_controller()
        
        self.setup_summary(self.m.scalar_tags, self.m.histogram_tags)
            
        tf.global_variables_initializer().run()
        
        mc_vars = list(self.mc_w.values()) + [self.mc_step_op]
        c_vars = list(self.c_w.values()) + [self.c_step_op]
        
        
        self._saver = tf.train.Saver(var_list = mc_vars + c_vars,
                                      max_to_keep=30)

        self.load_model()
        self.update_target_q_network(prefix = 'mc')
        self.update_target_q_network(prefix = 'c')
        
