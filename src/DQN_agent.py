import os
import random
import numpy as np
from functools import reduce
import tensorflow as tf
from metrics import Metrics

from base import Agent
from epsilon import Epsilon
from replay_memory import PriorityExperienceReplay, OldReplayMemory#, ReplayMemory
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
        memory_type = PriorityExperienceReplay if self.ag.pmemory else OldReplayMemory        
        self.memory = memory_type(config      = self.ag,
                                   screen_size = self.environment.state_size)       
        
       
        self.m = Metrics(self.config, self.logs_dir)       
        self.build_dqn()
        self.write_configuration()
        
    def train(self):
        self.flag_start_training = False
        self.start_step = self.step_op.eval()
        if self.ag.fresh_start:
            self.start_step = 0
        self.total_steps = self.ag.max_step + self.start_step# + self.ag.memory_size
        self.epsilon = Epsilon()
        self.epsilon.setup(self.ag, self.total_steps)
        old_obs = self.new_episode()

        self.m.start_timer()
                
        iterator = self.get_iterator(start_step  = self.start_step,
                                     total_steps = self.total_steps)
        
        for self.step in iterator:
            # 1. predict
            action = self.predict_next_action(old_obs)    
       
            # 2. act            
            info = {'is_SF'           : self.m.is_SF,
                    'display_episode' : self.display_episode,
                    'watch'           : self.gl.watch}
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
                self.m.close_episode()
                old_obs = self.new_episode()
             
            else:
                old_obs = new_obs.copy()
           
            if not self.is_testing_time(prefix = ''):
                continue

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
        else:
            ep = 1
        self.m.update_epsilon(value = ep)
        if random.random() < ep and not self.is_playing():
            action = random.randrange(self.environment.action_size)
        else:
            action = self.q_action.eval({self.s_t: [[old_obs]]})[0]

        return action

    def observe(self, old_screen, action, reward, screen, terminal):
        self.memory.add(old_screen, action, reward, screen, terminal)
        self.learn_if_ready(prefix = '')

    def q_learning_mini_batch(self):        
        (s_t, action, reward, s_t_plus_1, terminal), idx_list, p_list, \
                                        sum_p, count = self.memory.sample() 
        
        #assert all(reward < 9), str(reward)
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

        _, q_t, td_error, loss = self.sess.run([self.optim, self.q,
                                                             self.td_error,
                                             self.loss], feed_dict)
        if self.ag.pmemory:
            self.memory.update(idx_list, td_error)
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
                self.q, self.w['q_w'], self.w['q_b'] = utils.linear(last_layer,
                                                      self.environment.action_size,
                                                      name='q')
            
            self.q_action = tf.argmax(self.q, axis=1)
        self.create_target(self.ag)
        
        # optimizer
        self.build_optimizer(prefix = '')

        self.setup_summary(self.m.scalar_tags, self.m.histogram_tags)
        tf.global_variables_initializer().run()
        vars_ = list(self.w.values()) + [self.step_op]
        self._saver = tf.train.Saver(vars_, max_to_keep=30)

        self.load_model()
        self.update_target_q_network(prefix = '')
        
