class Epsilon():
    def __init__(self):
        pass
    
    def setup(self, ag, total_steps):
        """
        Sets up linear decay of epsilon
        """
        self.start = ag.ep_start
        self.end = ag.ep_end
        self.end_t = ag.ep_end_t_perc * total_steps
        
        
    def start_decaying(self, learn_start):
        """
        We let epsilon the moment from which we want the larning to happen
        hence epsilon will decay from learn_start onwards
        """
        self.learn_start = learn_start
        
        
        
    def steps_value(self, step):
        """
        Epsilon linear decay.
        Returns the epsilon value for a given step according to the setup
        """
        epsilon = (self.end + \
                   max(0., (self.start - self.end) \
                   * (self.end_t - max(0., step - self.learn_start)) \
                   / self.end_t))
        assert 0 <= epsilon <= 1, epsilon
        return epsilon
    
    def successes_value(self, successes, attempts):
        """
        Epsilon goal success decay
        Returns the epsilon value for a given number of successes / attempts
        ratio
        """
        epsilon = 1. - successes / (attempts + 1)
        
        assert epsilon > 0, str(epsilon) + ', '+ str(successes) + ', ' + str(attempts)

        return epsilon        
    
