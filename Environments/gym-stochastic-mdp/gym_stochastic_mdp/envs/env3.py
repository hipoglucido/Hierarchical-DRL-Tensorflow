import random
from mdp import MDP


TOTAL_STATES = 6
INITIAL_STATE = 1
TERMINAL_STATES = [0, TOTAL_STATES - 1]
TOTAL_ACTIONS = 2

BIG_REWARD = 1.
SMALL_REWARD = 1./ 100.
TRAP_STATES = [3, 4, 7]

RIGHT_FAILURE_PROB = 0.
class Trap_MDPEnv(MDP):
    
    def __init__(self):
        pass
  
    def configure(self, cnf):
        self.initial_state = cnf.initial_state
        super().configure(total_states = cnf.total_states,
                         total_actions = cnf.total_actions,
                         terminal_states = cnf.terminal_states)
        self.trap_states = TRAP_STATES
    
    def its_a_trap(self, state):
        return state in self.trap_states
        
    def step(self, action):
        if self.has_ended():
            raise RuntimeError("Environment already ended")
        
        assert(self.action_space.contains(action))
        
        info = dict()
        aux = 1 if action == 1 else -1
        transition = - aux if self.its_a_trap(self.current_state) else aux
        if transition and random.random() < RIGHT_FAILURE_PROB:
            transition = -1
            
        s = self.current_state + transition
        

        if self.state_space.contains(s):
            self.current_state = s
        else:
            #Don't move
            # e.g. top right state + right = top right state
            pass

                
        if not self.has_ended():
            reward, done = 0, False    
        elif self.current_state == self.state_size - 1:
            reward, done = BIG_REWARD, True
            
        else:
            reward, done = SMALL_REWARD, True
        one_hot_state = self.one_hot(self.current_state)
        
        return one_hot_state, reward, done, info
        

#Trap_MDPEnv().random_test()
    
    
    
    
    
    
    
    
