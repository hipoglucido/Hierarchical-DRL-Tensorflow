from __future__ import print_function
import os
import time
import random
import numpy as np
from tqdm import tqdm
from functools import reduce
import tensorflow as tf
import sys
from metrics import Metrics

from base import Agent, Epsilon
from history import History
from replay_memory import ReplayMemory, PriorityExperienceReplay, OldReplayMemory
from ops import linear, clipped_error, huber_loss, weighted_huber_loss
from utils import get_time, save_pkl, load_pkl

class DQNAgent(Agent):
    def __init__(self, config, environment, sess):
        super().__init__(config)
        self.sess = sess
        self.weight_dir = 'weights'

        self.config = config
        self.ag = self.config.ag
        self.gl = self.config.gl
        self.environment = environment
        self.ag.update({"q_output_length" : self.environment.action_size}, add = True)
        self.history = History(length_ = self.ag.history_length,
                               size    = self.environment.state_size)
        
      
        memory_type = PriorityExperienceReplay if self.ag.pmemory else OldReplayMemory
        
        self.memory = memory_type(config      = self.ag,
                                   model_dir  = self.model_dir,
                                   screen_size = self.environment.state_size)

        
            
        self.m = Metrics(self.config)
        
        self.build_dqn()
        self.config.print()
        self.write_configuration()
        
#        assert self.ag.memory_size < self.ag.max_step
   
    def train(self):
        start_step = 0    
        self.epsilon = Epsilon(self.ag, start_step)
        
        self.new_episode()

        self.m.start_timer()
        total_steps = self.ag.max_step + self.ag.memory_size
        if self.m.is_SF:   
            iterator = tqdm(range(start_step, total_steps),
                                                  ncols=70, initial=start_step)
        else:
            iterator = range(start_step, total_steps)
        
        print("\nFilling memory with %d random experiences..." % (self.ag.memory_size))
        for self.step in iterator:
            if self.memory.is_full() and self.step == self.ag.memory_size:
                print("\nLearning...")
#            if self.step == self.ag.learn_start:                
#                self.m.restart()
#            if self.memory.is_full():
#                self.m.restart()
            old_screen = self.history.get()
            # 1. predict
            action = self.predict_next_action()    
            
            # 2. act            
            screen, reward, terminal = self.environment.act(action)
            
            if self.m.is_SF:
                self.m.add_act(action, screen)
            else:
                self.m.add_act(action, self.environment.gym.one_hot_inverse(screen))
            if self.display_episode:
                self.console_print(action, reward)
                
                
                
            # 3. observe
            self.observe(old_screen, action, reward, screen, terminal)
            self.m.increment_external_reward(reward)
            
            if terminal:
                if self.display_episode:
                    self.console_print_terminal(reward)
                self.m.mc_step_reward = 0
                self.m.close_episode()
                self.new_episode()

            
#            if self.step < self.ag.learn_start:
#                continue
            if self.step % self.ag.test_step != self.ag.test_step - 1 or \
                                not self.memory.is_full():
                continue   
            self.m.compute_test(prefix = '', update_count = self.m.update_count)
            self.m.compute_state_visits()
#            self.m.print('')
            if self.m.has_improved(prefix = ''):
                self.step_assign_op.eval(
                        {self.step_input: self.step + 1})
     
#                self.save_model(self.step + 1)

                self.m.update_best_score()
                
           
                
            self.m.learning_rate = self.learning_rate_op.eval(
                            {self.learning_rate_step: self.step})
            summary = self.m.get_summary()
            self.m.filter_summary(summary)
            self.inject_summary(summary, self.step)
            self.write_output()
            if self.step > self.ag.max_step / 3:
                pass#
            self.m.restart()
            
    def new_episode(self):
        #screen, reward, action, terminal = self.environment.new_random_game()
        screen, _, _, _ = self.environment.new_game()        
        self.history.fill_up(screen)
        
        self.display_episode = random.random() < self.gl.display_prob and \
                                                    self.memory.is_full()
        
        return 
    
    def predict_next_action(self, test_ep = None):
        s_t = self.history.get()
        ep = test_ep or self.epsilon.steps_value(self.step)
        self.m.update_epsilon(value = ep)
        if random.random() < ep or not self.memory.is_full():
            action = random.randrange(self.environment.action_size)
        else:
            action = self.q_action.eval({self.s_t: [s_t]})[0]

        return action

    def observe(self, old_screen, action, reward, screen, terminal):
        #reward = max(self.min_reward, min(self.max_reward, reward)) #TODO understand
        # NB! screen is post-state, after action and reward
        
        assert np.sum(np.isnan(screen)) == 0, screen
#        if self.memory.is_full() and reward == -1:
#            print("_________________rr_____________________")
#            print("s_t\n",old_screen)
#            print("A",action)
#            print("s_t_plus_one\n",screen)
#            print("R", reward)
#            print("terminal", terminal + 0)
    #        
            
        #self.memory.add(screen, reward, action, terminal)
        self.memory.add(old_screen, action, reward, screen, terminal)
        self.history.add(screen)
        
        #if self.step > self.ag.learn_start:
        if self.memory.is_full():
            if self.step % self.ag.train_frequency == 0:
                self.q_learning_mini_batch()

            if self.step % self.ag.target_q_update_step == \
                                            self.ag.target_q_update_step - 1:
                self.update_target_q_network()


    def q_learning_mini_batch(self):
        if self.memory.count < self.history.length:
            return
        (s_t, action, reward, s_t_plus_1, terminal), idx_list, p_list, \
                                        sum_p, count = self.memory.sample()
#        s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()
#        print("______________________________________")
#        print("s_t\n",s_t[0].reshape(5,5))
#        print("A",action[0])
#        print("R", reward[0])
#        print("s_t_plus_1\n", s_t_plus_1[0].reshape(5,5))
#        print("terminal", terminal[0] + 0)
#        assert reward[0] in [0., -1.], reward[0]


#        print(s_t_plus_1.shape)
        
#        if self.config.ag.double_q:
#            
#            pred_action = self.q_action.eval({self.s_t: s_t_plus_1})
#            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
#            self.target_s_t: s_t_plus_1,
#            self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
#          })
#            target_q_t = (1. - terminal) * self.ag.discount * \
#                                        q_t_plus_1_with_pred_action + reward
#        else:
#            terminal = np.array(terminal) + 0.
#            q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})
#    
#            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
#            target_q_t = (1. - terminal) * self.ag.discount * max_q_t_plus_1 + reward
        target_q_t = self.generate_target_q_t(prefix       = '',
                                              reward       = reward,
                                              s_t_plus_1   = s_t_plus_1,
                                              terminal     = terminal)
        feed_dict = {
            self.target_q_t: target_q_t,
            self.action: action,
            self.s_t: s_t,
            self.learning_rate_step: self.step,
        }
        
        
        if self.ag.pmemory:
            beta = (1 - self.epsilon.steps_value(self.step)) + self.epsilon.end
            self.m.beta = beta
            loss_weight = (np.array(p_list)*count/sum_p)**(-beta)
            feed_dict[self.loss_weight] = loss_weight
#            print("___________")
#            print("loss_weight",loss_weight)
#            print("idx_list", idx_list)
#            print("p_list", p_list)
#            print("sum_p", sum_p)
#            print("count", count)
        _, q_t, td_error, loss, summary_str = self.sess.run([self.optim, self.q,
                                                             self.td_error,
                                             self.loss, self.q_summary], feed_dict)
        if self.ag.pmemory:
            self.memory.update(idx_list, td_error)
#       
#        
#        q_t_acted = np.array([q_t[i][a] for i, a in enumerate(action)])      
#        
#        td = np.abs(q_t_acted - target_q_t)
#        print("________________")
#        print(td_error)
            
        
        
        self.writer.add_summary(summary_str, self.step) #TODO what does this do?

        self.m.total_loss += loss
        self.m.total_q += q_t.mean()        
        self.m.update_count += 1
        self.m.td_error += td_error.mean()

        
        
        
    def build_dqn(self):
        self.w = {}
        
        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)

        # training network
        with tf.variable_scope('prediction'):
            
            # tf Graph input
            self.s_t = tf.placeholder("float",
                                    [None, self.ag.history_length,
                                      self.environment.state_size], name='s_t')
            print(self.s_t)
            shape = self.s_t.get_shape().as_list()
            self.s_t_flat = tf.reshape(self.s_t, [-1, reduce(
                                            lambda x, y: x * y, shape[1:])])
            
            last_layer = self.s_t_flat
            last_layer, histograms = self.add_dense_layers(architecture = self.ag.architecture,
                                               input_layer = last_layer,
                                               parameters = self.w,
                                               name_aux = '')
            if self.ag.dueling:
                self.q = self.add_dueling(prefix = '', input_layer = last_layer)
            else:
                self.q, self.w['q_w'], self.w['q_b'] = linear(last_layer,
                                                      self.environment.action_size,
                                                      name='q')
            
            self.q_action = tf.argmax(self.q, axis=1)
            
            q_summary = histograms
            avg_q = tf.reduce_mean(self.q, 0)
    
            print(avg_q)
            for idx in range(self.ag.q_output_length):
                print(idx, avg_q[idx])
                q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
            self.q_summary = tf.summary.merge(q_summary, 'q_summary')

        self.create_target(self.ag)


    
        # optimizer
        with tf.variable_scope('optimizer'):
            if self.ag.pmemory:
                self.loss_weight = tf.placeholder('float32', [None], name='loss_weight')
            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')

            action_one_hot = tf.one_hot(self.action, self.environment.action_size,
                                       1.0, 0.0, name = 'action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot,
                                   reduction_indices = 1, name = 'q_acted')
            self.td_error = tf.abs(self.target_q_t - q_acted)
            #delta = self.target_q_t - q_acted

            if self.ag.pmemory:
                self.loss = tf.reduce_mean(weighted_huber_loss(y_true = self.target_q_t,
                                                         y_pred = q_acted,
                                                            weights = self.loss_weight),
                                                          name='loss')
            else:
                self.loss = tf.reduce_mean(huber_loss(y_true = self.target_q_t,
                                                         y_pred = q_acted),
                                                          name='loss')
                
            self.learning_rate_step = tf.placeholder('int64', None, #*
                                            name='learning_rate_step')
            self.learning_rate_op = tf.maximum(#*
                    self.ag.learning_rate_minimum,
                    tf.train.exponential_decay(
                            learning_rate = self.ag.learning_rate,
                            global_step   = self.learning_rate_step,
                            decay_steps   = self.ag.learning_rate_decay_step,
                            decay_rate    = self.ag.learning_rate_decay,
                            staircase     = True))
            self.optim = tf.train.RMSPropOptimizer(
                                self.learning_rate_op, momentum=0.95,
                                epsilon=0.01).minimize(self.loss)
        
        self.setup_summary(self.m.scalar_tags, self.m.histogram_tags)
        tf.global_variables_initializer().run()
        vars_ = list(self.w.values()) + [self.step_op]
        self._saver = tf.train.Saver(vars_, max_to_keep=30)

        self.load_model()
        self.update_target_q_network()

    def update_target_q_network(self):
#        print("_____________________")
#        for name in self.w.keys():
#            print(name)
#            parameters = self.w[name].eval()
#            parameters_target = self.target_w[name].eval()
##            print(parameters)
##            print(parameters_target)
#            print(abs(parameters - parameters_target).sum())
        for name in self.w.keys():
#            print(name)
            parameters = self.w[name].eval()
#            parameters_target = self.target_w[name].eval()
#            print(parameters)
#            print("******")
#            print(parameters_target)
            self.target_w_assign_op[name].eval(
                            {self.target_w_input[name]: parameters})

    def save_weight_to_pkl(self):
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)

        for name in self.w.keys():
            save_pkl(self.w[name].eval(),
                            os.path.join(self.weight_dir, "%s.pkl" % name))

    def load_weight_from_pkl(self, cpu_mode=False):
        with tf.variable_scope('load_pred_from_pkl'):
            self.w_input = {}
            self.w_assign_op = {}

            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float32',
                            self.w[name].get_shape().as_list(), name=name)
                self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

        for name in self.w.keys():
            self.w_assign_op[name].eval({self.w_input[name]: load_pkl(
                            os.path.join(self.weight_dir, "%s.pkl" % name))})

        self.update_target_q_network()


    def play(self, n_step=10000, n_episode=100, test_ep=None, render=False):
        if test_ep == None:
            test_ep = self.ep_end

        test_history = History(self.config)

        if not self.display:
            gym_dir = '/tmp/%s-%s' % (self.env_name, get_time())
            self.env.env.monitor.start(gym_dir)

        best_reward, best_idx = 0, 0
        for idx in range(n_episode):
            screen, reward, action, terminal = self.env.new_random_game()
            current_reward = 0

            for _ in range(self.history_length):
                test_history.add(screen)

            for t in tqdm(range(n_step), ncols=70):
                # 1. predict
                action = self.predict(test_history.get(), test_ep)
                # 2. act
                screen, reward, terminal = self.env.act(action,
                                                        is_training=False)
                # 3. observe
                test_history.add(screen)

                current_reward += reward
                if terminal:
                    break

            if current_reward > best_reward:
                best_reward = current_reward
                best_idx = idx

            print("="*30)
            print(" [%d] Best reward : %d" % (best_idx, best_reward))
            print("="*30)

        if not self.display:
            self.env.env.monitor.close()
            #gym.upload(gym_dir, writeup='https://github.com/devsisters/DQN-tensorflow', api_key='')

"""
    def q_learning_mini_batch(self):
        if self.memory.count < self.history.length:
            return
        
        s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()
        
        q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})

        terminal = np.array(terminal) + 0.
        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
        target_q_t = (1. - terminal) * self.ag.discount * max_q_t_plus_1 + reward

        _, q_t, loss, summary_str = self.sess.run([self.optim, self.q,
                                             self.loss, self.q_summary], {
            self.target_q_t: target_q_t,
            self.action: action,
            self.s_t: s_t,
            self.learning_rate_step: self.step,
        })
        self.writer.add_summary(summary_str, self.step)

        self.m.total_loss += loss
        self.m.total_q += q_t.mean()        
        self.m.update_count += 1
"""