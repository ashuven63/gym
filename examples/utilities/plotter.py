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
        # print 'Loading pickle dump...'
        # print episode
        return episode

    def get_episode_stats(self, episode_num):
        steps = self.load_episode(episode_num)
        # print episodes
        env_states, agent_states = zip(*steps)
        env_states = list(env_states)
        agent_states = list(agent_states)
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
        # print episode_rewards
        mean_reward = np.mean(np.array(episode_rewards), axis=0)
        episodes_optimal_actions = []
        for episode_num in range(self.num_of_episodes):
            episode_optimal_actions = []
            sum_of_optimal_actions = 0
            episode_stat = episode_stats[episode_num]
            for i in range(self.num_of_steps):
                if episode_stat[1][i] == episode_stat[2][i]:
                    sum_of_optimal_actions += 1
                episode_optimal_actions.append(sum_of_optimal_actions * 100.0 / (i+1))
            episodes_optimal_actions.append(episode_optimal_actions)
        mean_optimal_actions = np.mean(np.array(episodes_optimal_actions),axis=0)
        plt.figure(1)
        ax1 = plt.subplot(211)
        ax1.set_ylim([0, 100])
        self.plot_reward(mean_reward)
        ax2 = plt.subplot(212)
        ax2.set_ylim([0, 100])
        self.plot_optimal_percent(mean_optimal_actions)
        plt.show()
