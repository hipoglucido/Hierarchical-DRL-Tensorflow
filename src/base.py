import os
import pprint
#from configuration import MetaControllerParameters, ControllerParameters
import inspect
import tensorflow as tf
import numpy as np
import utils
import ops
from functools import reduce
from ops import linear, clipped_error

pp = pprint.PrettyPrinter().pprint

class Epsilon():
    def __init__(self, config, start_step):
        self.start = config.ep_start
        self.end = config.ep_end
        self.end_t = config.ep_end_t
        
        self.learn_start = config.learn_start
        self.step = start_step
    
    def steps_value(self, step):
        epsilon = self.end + \
                max(0., (self.start - self.end) * \
                 (self.end_t -max(0., step - self.learn_start)) / self.end_t)
        assert epsilon > 0
        return epsilon
    
    def successes_value(self, successes, attempts):
        epsilon = 1. - successes / (attempts + 1)
        
        assert epsilon > 0, str(epsilon) + ', '+ str(successes) + ', ' + str(attempts)
#        print('99999',str(epsilon) + ', '+ str(successes) + ', ' + str(attempts))
        return epsilon        
    
    def mixed_value(self, successes, attempts):
        successes_value = self.successes_value(successes, attempts)
        #steps_value = self.steps_value(step)
        return max(successes_value, .1)
        
    
class Agent(object):
    """Abstract object representing an Reader model."""
    def __init__(self, config):
        self._saver = None
        self.config = config
        self.output = ''
    
    def add_output(self, txt):
        self.output += txt
    def console_print(self, action, reward):
        if self.m.is_hdqn:
            observation = self.c_history.get()[-1]
        else:
            observation = self.history.get()[-1]
     
        if self.environment.env_name == 'key_mdp-v0':
            out =  observation.reshape(self.environment.gym.shape) 
        else:
            out = self.environment.gym.one_hot_inverse(observation)
        msg = '\nS:\n' + str(out) + '\nA: ' + str(action) + '\nR: ' + str(reward)
        if self.m.is_hdqn:
            msg = msg + ', G: ' + str(self.current_goal.n)
        self.add_output(msg)
        print(msg)
    def console_print_terminal(self, reward):
        if self.m.is_hdqn:
            observation = self.c_history.get()[-1]
            perc = round(100 * self.c_step / self.c.max_step, 4)
            ep_r = self.mc_ep_reward
        else:
            observation = self.history.get()[-1]
            perc = round(100 * self.step / self.ag.max_step, 4)
            ep_r = self.m.ep_reward
        if self.environment.env_name == 'key_mdp-v0':
            out =  observation.reshape(self.environment.gym.shape) 
        else:
            out = self.environment.gym.one_hot_inverse(observation)
        msg = '\nS:\n' + str(out) + '\nEP_R: ' + str(ep_r)
        if reward == 1:
            msg += "\tSUCCESS"
        msg += "\n________________ " + str(perc) + "% ________________"[:150]
        self.add_output(msg)
        print(msg)
        
#        assert reward != 1
        

            
    def setup_summary(self, scalar_summary_tags, histogram_summary_tags):    
        """
        average.X   : mean X per step
        test.X      : total X per testing inverval
        episode.X Y : X Y per episode
        
        """        
        with tf.variable_scope('summary'):
            

            self.summary_placeholders = {}
            self.summary_ops = {}
            
            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder(
                                'float32', None, name=tag)
                self.summary_ops[tag]    = tf.summary.scalar("%s-/%s" % \
                        (self.environment.env_name, tag), self.summary_placeholders[tag])            

            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32',
                                         None, name=tag)
                self.summary_ops[tag]    = tf.summary.histogram(tag,
                                            self.summary_placeholders[tag])
            print(self.model_dir)
            print("Scalars: ", ", ".join(scalar_summary_tags))
            print("Histograms: ", ", ".join(histogram_summary_tags))
        self.writer = tf.summary.FileWriter('./logs/%s' % \
                                           self.model_dir, self.sess.graph)
         
    def inject_summary(self, tag_dict, step):

        summary_str_lists = self.sess.run(
                    [self.summary_ops[tag] for tag in tag_dict.keys()],
                    {self.summary_placeholders[tag]: value for tag, value \
                                                          in tag_dict.items()})
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, step)
    def show_attrs(self):
        import pprint
        attrs = vars(self).copy()
        try:
            del attrs['output']
        except:
            pass
        pprint.pprint(attrs)
    def add_dense_layers(self, architecture, input_layer, parameters, name_aux):
        #TODO delete config parameter
        last_layer = input_layer
        print(last_layer, "as input")
#        prefix = prefix + "_" if prefix != '' else prefix
#        
#        parameters = getattr(self, prefix + 'w')
        histograms = []
        for i, neurons in enumerate(architecture):
            number = 'l' + str(i + 1)
            layer_name = name_aux + number
            layer, weights, biases = \
                ops.linear(input_ = last_layer,
                       output_size = neurons,
                       activation_fn = tf.nn.relu,
                       name = layer_name)
            histograms += [tf.summary.histogram("w_" + layer_name, weights),
                           tf.summary.histogram("b_" + layer_name, biases)]
#                           tf.summary.histogram("o_" + layer_name, layer)]
            #setattr(self, layer_name, layer)
            parameters[layer_name + "_w"] = weights
            parameters[layer_name + "_b"] = biases
            last_layer = layer
#            print(layer_name, layer.get_shape().as_list(), 'added')        
            print(layer, 'added', layer_name)
        return last_layer, histograms

    def create_target(self, config):
        print("Creating target...")

        prefix = config.prefix + '_' if config.prefix != '' else config.prefix
        #config = config
        #config = self.config
#        # target network
        aux1 = prefix + 'target'                         # mc_target
        aux2 = aux1 + '_s_t'                             # mc_target_s_t
        aux3 = aux1 + '_w'                               # mc_target_w
        aux4 = aux1 + '_q'                               # mc_target_q
        aux5 = 'w' if prefix == '' else prefix + 'w'     # mc_w
        aux6 = aux4 + '_idx'                             # mc_target_q_idx        
        aux7 = aux4 + '_with_idx'                        # mc_target_q_with_idx
        aux8 = prefix + 'outputs_idx'                    # mc_outputs_idx
        target_w = {}
        
        
        setattr(self, aux3, target_w)
        with tf.variable_scope(aux1):
            target_s_t = tf.placeholder("float",
                        [None, config.history_length, self.environment.state_size],
                        name = aux2)
            shape = target_s_t.get_shape().as_list()
            target_s_t_flat = \
                tf.reshape(target_s_t,
                          [-1, reduce(lambda x, y: x * y, shape[1:])])
            if config.prefix == 'c':
                self.c_target_g_t = tf.placeholder("float",
                                   [None, self.ag.goal_size],
                                   name = 'c_target_g_t')
                self.target_gs_t = tf.concat([self.c_target_g_t, target_s_t_flat],
                                   axis = 1,
                                   name = 'c_target_gs_concat')
                last_layer = self.target_gs_t
            else:
                last_layer = target_s_t_flat
                
#            histograms_ = getattr(self, prefix + 'histograms')
            
            last_layer, _ = self.add_dense_layers(architecture = config.architecture,
                                               input_layer = last_layer,
                                               parameters = target_w,
                                               name_aux = prefix)
#            histograms_ += histograms
            
            
            if self.ag.dueling:
                print(aux4)
                target_q = self.add_dueling(prefix = aux1, input_layer = last_layer)
            else:
                target_q, weights, biases = \
                            linear(last_layer,
                                   config.q_output_length, name=aux4)  
                getattr(self, aux3)['q_w'] = weights
                getattr(self, aux3)['q_b'] = biases                   
            print(target_q)
            
            setattr(self, aux2, target_s_t)
            setattr(self, aux4, target_q)
            if self.config.ag.double_q:               
                #Double DQN                  
                target_q_idx = tf.placeholder('int32', [None, None], aux8)
                target_q_with_idx = tf.gather_nd(target_q, target_q_idx)
                setattr(self, aux6, target_q_idx)
                setattr(self, aux7, target_q_with_idx)
    
        self.show_attrs()
        with tf.variable_scope(prefix + 'pred_to_target'):
            target_w_input = {}
            target_w_assign_op = {}
            w = getattr(self, aux5)
            
            for name in w.keys():
#                print("__________________________")
                target_w_input[name] = tf.placeholder(
                               'float32',
                               target_w[name].get_shape().as_list(),
                               name=name)
                target_w_assign_op[name] = target_w[name].assign(
                                                value = target_w_input[name])
#                print(target_w_input[name])
#                print(target_w_assign_op[name])
        setattr(self, aux3 + "_input", target_w_input)
        setattr(self, aux3 + "_assign_op", target_w_assign_op)
        
        
        
    def save_model(self, step=None):
        print(" [*] Saving checkpoints...")

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, self.checkpoint_dir, global_step=step)

    def load_model(self):
        print(" [*] Loading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.checkpoint_dir, ckpt_name)
            self.saver.restore(self.sess, fname)
            print(" [*] Load SUCCESS: %s" % fname)
            return True
        else:
            print(" [!] Load FAILED: %s" % self.checkpoint_dir)
            return False
        
    def write_configuration(self):
        filename = self.model_dir + "_" + "cnf.txt"
        filepath = os.path.join(self.gl.logs_dir,self.model_dir, filename)
        with open(filepath, 'w') as fp:
            fp.write(self.config.to_str())
    def write_output(self):
        filename = self.model_dir + "_" + "episodes.txt"
        filepath = os.path.join(self.gl.logs_dir,self.model_dir, filename)
        with open(filepath, 'w') as fp:
            fp.write(self.output)
        
    @property
    def checkpoint_dir(self):
        return os.path.join('checkpoints', self.model_dir)

            
    @property
    def model_dir(self):
        chain = []
        for attr_fullname in self.config.gl.attrs_in_dir:
            [attr_type, attr_name] = attr_fullname.split('.')
            attr_value = getattr(getattr(self.config, attr_type), attr_name)
            if 'architecture' in attr_name:
                value = '-'.join([str(l) for l in attr_value])
            else:
                value = str(attr_value)
            attr_name_initials = ''.join([word[0] for word in attr_name.split('_')])
            part = attr_name_initials + str(value)
            chain.append(part)
        result = '_'.join(chain)
        return result

    @property
    def saver(self):
        if self._saver == None:
            self._saver = tf.train.Saver(max_to_keep=10)
        return self._saver




