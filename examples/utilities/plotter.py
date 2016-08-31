import matplotlib.pyplot as plt


class Plotter(object):
    def __init__(self, num_of_episodes, num_of_steps):
        self.num_of_episodes = num_of_episodes
        self.num_of_steps = num_of_steps

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

    def plot_episode_aggregate_results(self, rewards_list, opt_act_list):
        plt.figure(1)
        plt.subplot(211)
        for mean_reward in rewards_list:
            self.plot_reward(mean_reward)
        plt.subplot(212)
        for mean_optimal_actions in opt_act_list:
            self.plot_optimal_percent(mean_optimal_actions)
        plt.show()
