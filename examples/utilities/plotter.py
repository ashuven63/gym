import numpy as np
import matplotlib.pyplot as plt
import pickle


class Plotter(object):
    def __init__(self, outdir, num_of_episodes, num_of_steps):
        self.outdir = outdir
        self.num_of_episodes = num_of_episodes
        self.num_of_steps = num_of_steps

    def load_episode(self, episode_num):
        filename = '{0}/episode{1}'.format(self.outdir, episode_num)
        with open(filename, 'rb') as f:
            episode = pickle.load(f)
        print 'Loading pickle dump...'
        print episode
        return episode

    def get_episode_stats(self, episode_num):
        steps = (self.load_episode(episode_num)).values()
        # print episodes
        env_states, agent_states = zip(*steps)
        env_states = list(env_states)
        agent_states = list(agent_states)
        num_of_steps = len(steps)
        print 'length %s' % num_of_steps
        rewards = [agent_state['reward'] for agent_state in agent_states]
        optimal_action = [env_state['optimal_action'] for env_state in env_states]
        action_taken = [agent_state['action'] for agent_state in agent_states]
        return (rewards, optimal_action, action_taken)

    def plot_reward(self, rewards):
        plt.plot(rewards)
        plt.ylabel('Rewards')
        plt.xlabel('Time step')

    def plot_optimal_percent(self, optimal_percent):
        plt.plot(optimal_percent)
        plt.ylabel('Optimal Percent')
        plt.xlabel('Time step')

    def plot_single_episode(self, episode_num):
        rewards, _ = self.get_episode_stats(episode_num)
        self.plot_reward(rewards)

    def plot_episode_aggregate_results(self):
        episode_stats = [self.get_episode_stats(i) for i in range(self.num_of_episodes)]
        episode_rewards = [episode_stat[0] for episode_stat in episode_stats]
        print episode_rewards
        mean_reward = np.mean(np.array(episode_rewards), axis=0)
        episode_optimal_actions = []
        sum_of_optimal_actions = 0
        for i in range(self.num_of_steps):
            if episode_stat[1][i] == episode_stat[2][i]:
                sum_of_optimal_actions += 1
            episode_optimal_actions.append((sum_of_optimal_actions / self.num_of_steps))
        plt.figure(1)
        plt.subplot(211)
        self.plot_reward(mean_reward)
        plt.subplot(212)
        self.plot_optimal_percent(episode_optimal_actions)
        plt.show()
