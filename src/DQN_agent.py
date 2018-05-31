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
import time
from base import Agent, Epsilon
from history import History
from replay_memory import ReplayMemory, PriorityExperienceReplay, OldReplayMemory
from utils import linear, huber_loss, weighted_huber_loss
from utils import get_time, save_pkl, load_pkl
import utils
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
                                   screen_size = self.environment.state_size)

        
            
        self.m = Metrics(self.config)
        
        self.build_dqn()
#        if self.ag.mode == 'play':
#            self.gl.date = utils.get_timestamp() + "-" + self.gl.date
#        time.sleep(1)
#        self.config.print()
        self.write_configuration()
        
    def train(self):
        self.flag_start_training = False
        self.start_step = self.step_op.eval()
        if self.ag.fresh_start:
            self.start_step = 0
        total_steps = self.ag.max_step + self.start_step# + self.ag.memory_size
        self.epsilon = Epsilon()
        self.epsilon.setup(self.ag, total_steps)
        old_obs = self.new_episode()

        self.m.start_timer()
        
        if self.m.is_SF and self.gl.paralel == 0:   
            iterator = tqdm(range(self.start_step, total_steps),
                                                  ncols=70, initial=self.start_step)
        else:
            iterator = range(self.start_step, total_steps)
        
#        print("\nFilling memory with random experiences until step %d..." % \
#                                  (self.ag.learn_start))
        for self.step in iterator:
            # 1. predict
            action = self.predict_next_action(old_obs)    
       
            # 2. act            
            info = {'is_SF'           : self.m.is_SF,
                    'display_episode' : self.display_episode}
            new_obs, reward, terminal, info = self.environment.act(action, info)
            self.process_info(info)
           
            if self.m.is_SF:
                self.m.add_act(action)
            else:
                self.m.add_act(action, self.environment.gym.one_hot_inverse(new_obs))
            if self.display_episode:
                self.console_print(old_obs, action, reward)
            
                
            
            # 3. observe
            self.observe(old_obs, action, reward, new_obs, terminal)
            self.m.increment_external_reward(reward)
            
            if terminal:
                if self.display_episode:
                    self.console_print_terminal(reward, new_obs)
                #self.m.mc_step_reward = 0
                self.m.close_episode()
                old_obs = self.new_episode()
             
            else:
                old_obs = new_obs.copy()
           
            if not self.is_testing_time(prefix = ''):
                continue
#            if not self.is_ready_to_learn(prefix = ''):
#                #Monitor shouldn't start if learning hasn't
#                continue
#            if self.step % self.ag.test_step != self.ag.test_step - 1:# or \
#                                #not self.memory.is_full():
#                continue   
            self.m.compute_test(prefix = '', update_count = self.m.update_count)
            self.m.compute_state_visits()
            
            if self.m.has_improved():
                self.step_assign_op.eval(
                        {self.step_input: self.step + 1})
     
                self.save_model(self.step + 1)

                self.m.update_best_score()
                
            self.send_some_metrics(prefix = '')
            summary = self.m.get_summary()
            self.m.filter_summary(summary)
            self.inject_summary(summary, self.step)
            self.write_output()
            
            self.m.restart()
            

    
    def predict_next_action(self, old_obs):
        
        if self.is_ready_to_learn(prefix = ''):
            ep = self.epsilon.steps_value(self.step)
#            print("______")
#            print("step",self.step)
#            print("learn_start",self.epsilon.learn_start)
#            print("start", self.epsilon.start)
#            print("end",self.epsilon.end)
#            print("end_t",self.epsilon.end_t)
#            print("ep",ep)
        else:
            ep = 1
        self.m.update_epsilon(value = ep)
        if random.random() < ep:
            action = random.randrange(self.environment.action_size)
        else:
            action = self.q_action.eval({self.s_t: [[old_obs]]})[0]

        return action

    def observe(self, old_screen, action, reward, screen, terminal):
        #reward = max(self.min_reward, min(self.max_reward, reward)) #TODO understand
        # NB! screen is post-state, after action and reward
        
#        assert np.sum(np.isnan(screen)) == 0, screen
#        if self.memory.is_full() and reward == -1:
#            print("_________________rr_____________________")
#            print("s_t\n",old_screen.reshape(3,3))
#            print("A",action)
#            print("s_t_plus_one\n",screen.reshape(3,3))
#            print("R", reward)
#            print("terminal", terminal + 0)
            
            
        #self.memory.add(screen, reward, action, terminal)
        self.memory.add(old_screen, action, reward, screen, terminal)
        #self.history.add(screen)
        self.learn_if_ready('')
#        if self.is_ready_to_learn(prefix = ''):
#            if not self.flag_start_training:
#                print("\nLearning started at step %d with %d experiences in memory"\
#                                      % (self.step, self.memory.count))
#                self.flag_start_training = True
#            if self.step % self.ag.train_frequency == 0:
#                self.q_learning_mini_batch()
#
#            if self.step % self.ag.target_q_update_step == \
#                                            self.ag.target_q_update_step - 1:
#                self.update_target_q_network(prefix = '')


    def q_learning_mini_batch(self):
      
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
            
        _, q_t, td_error, loss = self.sess.run([self.optim, self.q,
                                                             self.td_error,
                                             self.loss], feed_dict)
        if self.ag.pmemory:
            self.memory.update(idx_list, td_error)
#       
#        
#        q_t_acted = np.array([q_t[i][a] for i, a in enumerate(action)])      
#        
#        td = np.abs(q_t_acted - target_q_t)
#        print("________________")
#        print(td_error)
            
        
        
        #self.writer.add_summary(summary_str, self.step) #TODO what does this do?

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
            
#            q_summary = histograms
#            avg_q = tf.reduce_mean(self.q, 0)
#    
#            print(avg_q)
#            for idx in range(self.ag.q_output_length):
#                print(idx, avg_q[idx])
#                q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
#            self.q_summary = tf.summary.merge(q_summary, 'q_summary')

        self.create_target(self.ag)


        
        # optimizer
        self.build_optimizer(prefix = '')

        self.setup_summary(self.m.scalar_tags, self.m.histogram_tags)
        tf.global_variables_initializer().run()
        vars_ = list(self.w.values()) + [self.step_op]
        self._saver = tf.train.Saver(vars_, max_to_keep=30)

        self.load_model()
        self.update_target_q_network(prefix = '')
        

        
#        assert self.ag.memory_size < self.ag.max_step
#    def play(self):
#        self.start_step = self.step_op.eval()
#        
#        
#        old_obs = self.new_episode()
#
#        self.m.start_timer()
#        self.step = 0
#        while 1:
#
#            # 1. predict
#            action = self.predict_next_action(old_obs)    
#       
#            # 2. act            
#            new_obs, reward, terminal = self.environment.act(action)
#           
#            if self.m.is_SF:
#                self.m.add_act(action)
#            else:
#                self.m.add_act(action, self.environment.gym.one_hot_inverse(new_obs))
#            if self.display_episode:
#                self.console_print(old_obs, action, reward)
#            
#                
#            
#            # 3. observe
#            self.observe(old_obs, action, reward, new_obs, terminal)
#            self.m.increment_external_reward(reward)
#            
#            if terminal:
#                if self.display_episode:
#                    self.console_print_terminal(reward, new_obs)
#                #self.m.mc_step_reward = 0
#                self.m.close_episode()
#                old_obs = self.new_episode()
#             
#            else:
#                old_obs = new_obs.copy()
#           
#            
#            self.step += 1
#            if self.step % self.ag.test_step != self.ag.test_step - 1:
#                continue   
#            
#            self.m.compute_test(prefix = '', update_count = 0)
#            self.m.compute_state_visits()
#            
#
#            summary = self.m.get_summary()
#            self.m.filter_summary(summary)
#            self.inject_summary(summary, self.step)
#            self.write_output()
#            
#            self.m.restart()
        