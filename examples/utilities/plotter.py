import numpy as np
import matplotlib.pyplot as plt
import pickle

class Plotter(object):

	def __init__(self, outdir, num_of_episodes):
		self.outdir = outdir
		self.num_of_episodes = num_of_episodes

	def load_episode(self, episode_num):
		filename = '{0}/episode{1}'.format(self.outdir, episode_num)
		episode = pickle.load( open( filename, "rb" ) )
		return episode

	def get_episode_stats(self, episode_num):
		episodes = (self.load_episode(episode_num)).values()
		#print episodes
		env_states, agent_states = zip(*episodes)
		env_states = list(env_states)
		agent_states = list(agent_states)
		num_of_steps = len(episodes)
		print 'length %s' % num_of_steps
		rewards = [agent_state['reward'] for agent_state in agent_states]
		optimal_actions = [env_state['optimal_value'] for env_state in env_states]
		return (rewards, optimal_actions)

	def plot_reward(self, rewards):
		plt.plot(rewards)
		plt.ylabel('Rewards')
		plt.xlabel('Time step')
		plt.show()

	def plot_single_episode(self, episode_num):
		rewards, _ = self.get_episode_stats(episode_num)
		self.plot_reward(rewards)
		# TODO: Graph for optimal actions. Currently just plotting the rewards. 

	def plot_episode_aggregate_results(self):
		aggregate_results = [self.get_episode_stats(i) for i in range(self.num_of_episodes)]
		aggregate_reward = [i[0] for i in aggregate_results]
		mean_reward = np.mean(np.array(aggregate_reward), axis=0)
		self.plot_reward(mean_reward)













