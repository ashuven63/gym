import matplotlib.pyplot as plt


class Plotter(object):
    def __init__(self, num_of_episodes, num_of_steps):
        self.num_of_episodes = num_of_episodes
        self.num_of_steps = num_of_steps

    def plot_reward(self, rewards, label):
        reward_handle, = plt.plot(rewards, label=label)
        plt.ylabel('Rewards')
        plt.xlabel('Time step')
        return reward_handle

    def plot_optimal_percent(self, optimal_percent, label):
        opt_perc_handle, = plt.plot(optimal_percent, label=label)
        plt.ylabel('Optimal Percent')
        plt.xlabel('Time step')
        return opt_perc_handle

    def plot_single_episode(self, episode_num):
        rewards, _ = self.get_episode_stats(episode_num)
        self.plot_reward(rewards, 'one_ep')

    def plot_episode_aggregate_results(self, rewards_list, opt_act_list, label_list):
        plt.figure(1)
        plt.subplot(211)
        handles = []
        for mean_reward, label in zip(rewards_list, label_list):
            handles.append(self.plot_reward(mean_reward, label))
        plt.legend(handles=handles, loc=4)

        plt.subplot(212)
        handles = []
        for mean_optimal_actions, label in zip(opt_act_list, label_list):
            handles.append(self.plot_optimal_percent(mean_optimal_actions, label))
        plt.legend(handles=handles, loc=4)

        plt.show()
