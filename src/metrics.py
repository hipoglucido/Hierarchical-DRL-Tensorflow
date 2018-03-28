# -*- coding: utf-8 -*-
import numpy as np
import time

class Metrics:
	def __init__(self, config, goals):
		self.mc_max_avg_ep_reward = 0
		self.c_max_avg_ep_reward = 0 #Not used for now
		self.define_metrics(goals)
		self.restart()
		self.mc_params = config.mc_params
		self.c_params = config.c_params
		

	def define_metrics(self, goals):
		self.scalar_global_tags = ['elapsed_time', 'games']
		self.mc_scalar_tags = ['mc_step_reward']
		self.c_scalar_tags = []
		
		
		self.scalar_dual_tags = ['avg_reward', 'avg_loss', 'avg_q', \
					 'max_ep_reward', 'min_ep_reward', \
					 'avg_ep_reward', 'learning_rate', 'total_reward', \
					 'ep_reward', 'total_loss', 'total_q', 'update_count']
		for tag in self.scalar_dual_tags:
			self.mc_scalar_tags.append("mc_" + tag)
			self.c_scalar_tags.append("c_" + tag)
			
		
		self.histogram_global_tags = []
		self.mc_histogram_tags = ['mc_goals', 'mc_ep_rewards']
		self.c_histogram_tags = ['c_actions', 'c_ep_rewards']	
		
		self.goal_tags = []
		for k, goal in goals.items():
			name = goal.name
			successes_tag = name + '_successes'
			frequency_tag = name + '_freq'
			success_rate_tag = name + '_success_rate'
			self.goal_tags += [successes_tag, frequency_tag, success_rate_tag]
			 
		self.mc_scalar_tags += self.goal_tags
		self.scalar_tags = self.mc_scalar_tags + self.c_scalar_tags + self.scalar_global_tags
		self.histogram_tags = self.histogram_global_tags + self.mc_histogram_tags + \
										self.c_histogram_tags
	def restart(self):
		
		for s in self.scalar_tags:
#			print(s)
			setattr(self, s, 0.)
		for h in self.histogram_tags:
#			print(h)
			setattr(self, h, [])
		
		try:
			self.restart_timer()
		except AttributeError:
			self.start_timer()
		
			
	def store_goal_result(self, goal, achieved):
		name = goal.name
		frequency_tag = name + '_freq'
		setattr(self, frequency_tag, getattr(self, frequency_tag) + 1.)
		if achieved:
			successes_tag = name + '_successes'
			setattr(self, successes_tag, getattr(self, successes_tag) + 1.)
			
	def compute_goal_results(self, goals):
		
		for _, goal in goals.items():
			name = goal.name
			successes_tag = name + '_successes'
			frequency_tag = name + '_freq'
			try:
				success_rate = getattr(self, successes_tag) / \
												getattr(self, frequency_tag)
			except ZeroDivisionError:
				
				success_rate = 0
			setattr(self, name + '_success_rate', success_rate)
#			print(name + '_success_rate', getattr(self, name + '_success_rate'))
	def update_best_score(self):
		self.mc_max_avg_ep_reward = max(self.mc_max_avg_ep_reward,
									  self.mc_avg_ep_reward)
		self.c_max_avg_ep_reward = max(self.c_max_avg_ep_reward,
									  self.c_avg_ep_reward)
		
	def has_improved(self, prefix):
		return self.mc_max_avg_ep_reward * 0.9 <= self.mc_avg_ep_reward
	
	
	def print(self, prefix):
		prefix = prefix + "_"
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

		
	def increment_rewards(self, internal, external):
		self.c_total_reward += internal
		self.c_ep_reward += internal
		
		self.mc_total_reward += external		
		self.mc_ep_reward += external
		self.mc_step_reward += external
		
		

	def filter_summaries(self, *args):
		
		exclude_inside = ['successes']
		exclude_equals = []
		
		for summary in args:
			for key in summary:
				if any([key in ei for ei in exclude_inside]) or \
								 any([key in ee for ee in exclude_equals]):
				   del summary[key]
			   
			
	def get_summaries(self):
		
		mc_summary = {}
		c_summary = {}
		global_summary = {}		
		
		for global_tag in self.scalar_global_tags:
			global_summary[global_tag] = getattr(self, global_tag)
			
		for mc_scalar_tag in self.mc_scalar_tags:
			mc_summary[mc_scalar_tag] = getattr(self, mc_scalar_tag)
			
		for c_scalar_tag in self.c_scalar_tags:
			c_summary[c_scalar_tag] = getattr(self, c_scalar_tag)
		
		#self.filter_summaries(global_summary, mc_summary, c_summary)
		print("Global",global_summary)
		print("MC",mc_summary)
		print("C",c_summary)
		return global_summary, mc_summary, c_summary
		
		
	
	
	def compute_test(self, prefix, update_count):
		assert prefix in ['c', 'mc']
		prefix = prefix + '_'
		config = getattr(self, prefix + 'params')
		test_step = config.test_step
		total_reward = getattr(self, prefix + 'total_reward')
		setattr(self, prefix + 'avg_reward', total_reward / test_step)
		total_loss = getattr(self, prefix + 'total_loss')
		setattr(self, prefix + 'avg_loss', total_loss / update_count)
		total_q = getattr(self, prefix + 'total_q')
		setattr(self, prefix + 'avg_q', total_q / update_count)
		
		ep_rewards = getattr(self, prefix + 'ep_rewards')
		
		try:
			setattr(self, prefix + 'max_ep_reward', np.max(ep_rewards))
			setattr(self, prefix + 'min_ep_reward', np.min(ep_rewards))
			setattr(self, prefix + 'avg_ep_reward', np.mean(ep_rewards))
		except Exception as e:
			print(str(e))
			for s in ['max', 'min', 'avg']:
				setattr(self, prefix + s +'_ep_reward', 0.)
		
				
	def mc_add_update(self, loss, q):
		self.mc_total_loss += loss
		self.mc_total_q += q
		self.mc_update_count += 1
	
	def c_add_update(self, loss, q):
		self.c_total_loss += loss
		self.c_total_q += q		
		self.c_update_count += 1

		
		
	def close_episode(self):
		self.games += 1
		
		self.mc_ep_rewards.append(self.mc_ep_reward)
		self.c_ep_rewards.append(self.c_ep_reward)
		self.mc_ep_reward, self.c_ep_reward = 0., 0.