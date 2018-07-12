import os
import random
import numpy as np
from functools import reduce
import tensorflow as tf


import base
from replay_memory import PriorityExperienceReplay, OldReplayMemory#, ReplayMemory
import utils
from goals import MDPGoal, generate_SF_goals
from metrics import Metrics
from constants import Constants as CT
from epsilon import Epsilon
        
class HDQNAgent(base.Agent):
    def __init__(self, config, environment, sess):
        super().__init__(config)
        self.sess = sess
        self.weight_dir = 'weights'

        self.config = config
        #self.ag = self.config.ag
        self.c_ag = self.config.ag.c
        self.mc_ag = self.config.ag.mc
        self.gl = self.config.gl
        self.environment = environment
        self.goals = self.define_goals()
        
        self.mc_ag.update({"q_output_length" : self.goal_size}, add = True)
        self.c_ag.update({"q_output_length" : self.environment.action_size}, add = True)
        
        
#        memory_type = PriorityExperienceReplay if self.ag.pmemory else OldReplayMemory
#        if self.mc.pmemory:
#        self.mc_memory = memory_type(config       = self.mc_ag,
#                                      screen_size  = self.environment.state_size)
#        self.c_memory = memory_type(config        = self.c_ag,
#                                      screen_size  = self.environment.state_size + \
#                                                          self.goal_size)   
        self.mc_memory = self.create_memory(config = self.mc_ag,
                                         size   = self.environment.state_size)
        self.c_memory = self.create_memory(config = self.c_ag,
                                         size   = self.environment.state_size + \
                                                          self.goal_size)
       
        self.m = Metrics(self.config, self.logs_dir, self.goals)
    
        #self.config.print()
        self.build_hdqn()
        self.write_configuration()

        
    def get_goal(self, n):
        return self.goals[n]
        
    def define_goals(self):
        
        if self.environment.env_name in CT.MDP_envs:
            self.goal_size = self.environment.state_size
            goals = {}
            for n in range(self.goal_size):
                goal_name = "g" + str(n)
                goal = MDPGoal(n, goal_name, self.c_ag)            
                goal.setup_one_hot(self.goal_size)
                goals[goal.n] = goal
        elif self.environment.env_name in CT.SF_envs:
            #Space Fortress
            goal_names = \
                CT.goal_groups[self.environment.env_name][self.config.ag.goal_group]
            goals = generate_SF_goals(
                    environment = self.environment,
                    goal_names  = goal_names)
            self.goal_size = len(goals)
            
        else:
            raise ValueError("No prior goals for " + self.environment.env_name)
        self.goal_probs = np.array([1 / len(goals) for _ in goals])
        return goals
    
    def set_next_goal(self, obs):
        
        if self.is_ready_to_learn(prefix = 'mc'):
            ep = self.mc_epsilon.steps_value(self.c_step)


        else:
            ep = 1
        
        self.m.update_epsilon(value = ep)
        if random.random() < ep and not self.is_playing():
            
            n_goal = random.randrange(self.goal_size)
        else:
            
            n_goal = self.mc_q_action.eval({self.mc_s_t: [[obs]]})[0]
        self.mc_old_obs = obs
        self.m.mc_goals.append(n_goal)
        goal = self.get_goal(n_goal)
        goal.set_counter += 1
        self.current_goal = goal
        self.current_goal.achieved_inside_frameskip = False
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
        self.mc_memory.add(self.mc_old_obs, goal_n, ext_reward, new_obs, terminal)
        

    def c_observe(self, old_obs, action, int_reward, new_obs, terminal):
        if self.is_playing():
            return
        old_state = np.hstack([self.current_goal.one_hot, old_obs])
        next_state = np.hstack([self.current_goal.one_hot, new_obs])
        self.c_memory.add(old_state, action, int_reward, next_state, terminal)
        #Update C
        self.learn_if_ready(prefix = 'c')
        #Update MC   
        self.learn_if_ready(prefix = 'mc')
        
    def mc_q_learning_mini_batch(self):      
        #Sample batch from memory
        (s_t, goal, ext_reward, s_t_plus_1, terminal), idx_list, p_list, \
                                        sum_p, count = self.mc_memory.sample()

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
        
        if self.mc_ag.pmemory:
            # Prioritized replay memory
            beta = (1 - self.mc_epsilon.steps_value(self.c_step)) + self.mc_epsilon.end        
            self.m.mc_beta = beta
            loss_weight = (np.array(p_list)*count/sum_p)**(-beta)
            feed_dict[self.mc_loss_weight] = loss_weight  
            

        _, q_t, mc_td_error, loss = self.sess.run([self.mc_optim,
                                                             self.mc_q,
                                                             self.mc_td_error,
                                                             self.mc_loss],
                                                             #self.mc_q_summary],
                                                            feed_dict)

        self.m.mc_add_update(loss, q_t.mean(), mc_td_error.mean())
        
    def c_q_learning_mini_batch(self):
        #Sample batch from memory        
        (s_t, action, int_reward, s_t_plus_1, terminal), idx_list, p_list, \
                                        sum_p, count = self.c_memory.sample()
        
        #TODO: optimize how goals are stored in memory (and how are they retrieved)
        g_t = np.vstack([g[0] for g in s_t[:, :, :self.goal_size]]) 
        s_t = s_t[:, :, self.goal_size:]
        
        
        g_t_plus_1 = np.vstack([g[0] for g in s_t_plus_1[:, :, :self.goal_size]])
        s_t_plus_1 = s_t_plus_1[:, :, self.goal_size:]
        
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
        if self.c_ag.pmemory:
            if self.is_ready_to_learn(prefix = 'mc'):
                beta = (1 - self.mc_epsilon.steps_value(self.c_step)) + \
                                                        self.mc_epsilon.end
            else:
                beta = self.mc_epsilon.end
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
        self.m.c_add_update(loss, q_t.mean(), c_td_error.mean())


    def play(self):
        self.train()
    
    def train(self):
        self.start_train_timer()
        #Auxiliary flags (for monitoring)
        self.mc_flag_start_training, self.c_flag_start_training = False, False
        self.c_learnt = False  #Indicates if C already knows how to achieve goals
        
        #Initial step (only useful when resuming training)
        self.mc_start_step = self.mc_step_op.eval()
        self.c_start_step = self.c_step_op.eval()
        
        #Total steps for the session
        self.total_steps = self.config.ag.max_step + self.c_start_step   
        
        #Set up epsilon ofr MC (e-greedy, linear decay)
        self.mc_epsilon = Epsilon()
        self.mc_epsilon.setup(self.mc_ag, self.total_steps)
        old_obs = self.new_episode()        
        self.m.start_timer()
        
        # Initial goal
        self.mc_step = self.mc_start_step
        self.c_step = self.c_start_step
        self.set_next_goal(old_obs)
        
        
        iterator = self.get_iterator(start_step  = self.c_start_step,
                                     total_steps = self.total_steps)
            
        for self.c_step in iterator:

            # Controller acts
            action = self.predict_next_action(old_obs)
            info = {'goal_name'       : self.current_goal.name,
                    'is_SF'           : self.m.is_SF,
                    'display_episode' : self.display_episode,
                    'watch'           : self.gl.watch,
                    'goal'            : self.current_goal}
            
            new_obs, ext_reward, terminal, info = self.environment.act(
                                        action = action,
                                        info   = info)
            self.process_info(info)            
            self.m.add_act(action, self.environment.gym.one_hot_inverse(new_obs))
            

            goal_achieved = self.current_goal.is_achieved(new_obs, action, info)
            int_reward = 1. if goal_achieved else 0.
            int_reward -= self.c_ag.intrinsic_time_penalty
            self.c_observe(old_obs, action, int_reward, new_obs,
                                       terminal or goal_achieved)
            
            if self.display_episode:
                self.console_print(old_obs, action, ext_reward, int_reward)  
            self.m.increment_rewards(int_reward, ext_reward)

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

                else:
                    old_obs = new_obs.copy()
                """
                self.mc_step is the MC equivalent for self.c_step (global counter)
                and self.m_mc_steps is the MC equivalent for self.c.test_step,
                which is constant in the case of C but not in the case of MC
                """
                self.mc_step += 1   
                self.m.mc_steps += 1
                
                # Meta-controller sets goal
                self.set_next_goal(old_obs)
                self.m.mc_goals.append(self.current_goal.n)
            if not terminal:
                old_obs = new_obs.copy()

            if not self.is_testing_time(prefix = 'c'):
                continue
            
            self.m.compute_test('c')
            self.m.compute_test('mc')
            goal_success_rate = self.m.compute_goal_results(self.goals)
            self.c_learnt = goal_success_rate > self.c_ag.learnt_threshold

            self.m.compute_state_visits()
            
            if self.m.has_improved():
                self.c_step_assign_op.eval(
                        {self.c_step_input: self.c_step + 1})
                self.mc_step_assign_op.eval(
                        {self.mc_step_input: self.mc_step + 1})
                self.delete_last_checkpoints()
                self.save_model(self.c_step + 1)
                #self.save_model(self.mc_step + 1)
                self.m.update_best_score()
                

            self.send_some_metrics(prefix = 'mc')
            self.send_some_metrics(prefix = 'c')
            summary = self.m.get_summary()
            self.m.filter_summary(summary)
            #self.m.rename_summary(summary)
            self.inject_summary(summary, self.c_step)
            self.write_output()
            # Restart metrics
            self.m.restart()
        self.stop_train_timer()
            
        
        
    def build_meta_controller(self):
        self.mc_w = {}
        self.mc_target_w = {}
        # training meta-controller network
        with tf.variable_scope('mc_prediction'):
            
            self.mc_s_t = tf.placeholder("float",
                        [None, 1, self.environment.state_size],
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
            if self.mc_ag.dueling:
                self.mc_q = self.add_dueling(prefix = 'mc',
                                             input_layer = last_layer)
            else:
                self.mc_q, self.mc_w['q_w'], self.mc_w['q_b'] = utils.linear(
                                                          last_layer,
                                                          self.goal_size,
                                                          name='mc_q')
            self.mc_q_action= tf.argmax(self.mc_q, axis=1)
            
            q_summary = histograms
            avg_q = tf.reduce_mean(self.mc_q, 0)
            
            for idx in range(self.mc_ag.q_output_length):
                q_summary.append(tf.summary.histogram('mc_q/%s' % idx, avg_q[idx]))
            self.mc_q_summary = tf.summary.merge(q_summary, 'mc_q_summary')

        # target network
        self.create_target(prefix = 'mc')

        #Meta Controller optimizer
        self.build_optimizer(prefix = 'mc')

        
        
    def build_controller(self):
        self.c_w = {}
        self.c_target_w = {}
    
        with tf.variable_scope('c_prediction'):
            #input_size = self.environment.state_size + self.goal_size
            
            self.c_s_t = tf.placeholder("float",
                                [None, 1,
                                self.environment.state_size],
                                name = 'c_s_t')
            shape = self.c_s_t.get_shape().as_list()
            self.c_s_t_flat = tf.reshape(self.c_s_t, [-1, reduce(
                    lambda x, y: x * y, shape[1:])])
            self.c_g_t = tf.placeholder("float",
                               [None, self.goal_size],
                               name = 'c_g_t')
            self.c_gs_t = tf.concat([self.c_g_t, self.c_s_t_flat],
                           axis = 1,
                           name = 'c_gs_concat')
            last_layer = self.c_gs_t
            last_layer, histograms = self.add_dense_layers(
                                            architecture = self.c_ag.architecture,
                                               input_layer = last_layer,
                                               parameters = self.c_w,
                                               name_aux= '')
            if self.c_ag.dueling:
                self.c_q = self.add_dueling(prefix = 'c',
                                            input_layer = last_layer)
            else:
                self.c_q, self.c_w['q_w'], self.c_w['q_b'] = \
                                      utils.linear(last_layer,
                                      self.environment.action_size,
                                      name='c_q')
            self.c_q_action= tf.argmax(self.c_q, axis=1)
            
            q_summary = histograms
            avg_q = tf.reduce_mean(self.c_q, 0)
            

            for idx in range(self.c_ag.q_output_length):
                q_summary.append(tf.summary.histogram('c_q/%s' % idx, avg_q[idx]))
            self.c_q_summary = tf.summary.merge(q_summary, 'c_q_summary')

        # target network
        self.create_target(prefix = 'c')
        
        
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
                

        self.build_meta_controller()

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
        
