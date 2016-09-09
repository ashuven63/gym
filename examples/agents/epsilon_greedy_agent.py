import random
import logging
import sys
import gym
import os
import numpy as np
import shutil

sys.path.append("../utilities")
from logger import Logger
from plotter import Plotter
from result_parser import ResultParser


class EpsilonGreedyAgent(object):
    """Expects the driver to call update after every call to act"""
    def __init__(self, action_space, epsilon=0.0, alpha=0.1, recency_weighting=False, init_value=0):
        self.action_space = action_space
        self.epsilon = epsilon
        self.alpha = alpha
        self.recency_weighting = recency_weighting
        self.init_value = init_value
        self.reset()

    # TODO(klad) : Handle observations for generic case.
    def act(self, observation):
        """Return next action to take and update prev_action"""
        if np.random.uniform() < self.epsilon:
            # Select a non optimal policy randomly.
            action = self.action_space.sample()
        else:
            # Select optimal action with random tie breaking.
            action = self.optimal_action()
        self.prev_action = action
        return action

    def optimal_action(self):
        """Find optimal action."""
        best_value = max([self.Q[action] for action in range(0, self.action_space.n)])
        # print 'best_value = ', best_value
        optimal_actions = [action for action in self.Q if self.Q[action] == best_value]
        # print 'Optimal actions are ', optimal_actions
        assert (len(optimal_actions) > 0)
        # Return randomly from optimal actions.
        return random.choice(optimal_actions)

    def update(self, reward):
        """Update action estimates after receiving reward from environment"""
        self.cur_reward = reward
        self.N[self.prev_action] += 1
        if not self.recency_weighting:
            self.Q[self.prev_action] += (reward - self.Q[self.prev_action]) / self.N[self.prev_action]
        else:
            self.Q[self.prev_action] += self.alpha * (reward - self.Q[self.prev_action])

    def get_state(self):
        state = {'epsilon': self.epsilon,
                 'alpha': self.alpha,
                 'recency_weighting': self.recency_weighting,
                 'init_value': self.init_value,
                 'action': self.prev_action,
                 'reward': self.cur_reward,
                 'Q': self.Q,
                 'N': self.N}
        return state

    def reset(self):
        self.prev_action = None
        self.cur_reward = None
        # Initialize Q estimates and counts for actions
        self.Q = {action: self.init_value for action in range(self.action_space.n)}
        self.N = {action: 0 for action in range(self.action_space.n)}

    def __str__(self):
        print 'Action\tEstimate\tFrequency'
        for action in self.Q:
            print '{0}\t{1}\t{2}'.format(action, self.Q[action], self.N[action])
        print 'Action{0}'.format(self.prev_action)
        print 'Reward{0}'.format(self.cur_reward)
        return ""


def run_expt(expt, logger):
    env, agents, episode_count, max_steps, outdirs, labels = expt
    for agent, outdir in zip(agents, outdirs):
        logger.info('Now running the next agent')
        for i in range(episode_count):
            ob = env.reset()
            agent.reset()
            state_logger = Logger(env, agent)
            for j in range(max_steps):
                action = agent.act(ob)
                ob, reward, done, _ = env.step(action)
                agent.update(reward)
                state_logger.update_log(j)
            state_logger.dump_log('{0}/episode{1}'.format(outdir, i))
            if i % 10 == 0:
                print 'Completed ', i, ' episodes'
        logger.info("Successfully ran EpsilonGreedyAgent")


def parse_results(expt, logger):
    logger.info("Parsing the results")
    _, _, episode_count, max_steps, outdirs, _ = expt
    rewards_list, opt_act_list = [], []
    for outdir in outdirs:
        parser = ResultParser(outdir, episode_count, max_steps)
        [mean_reward, mean_optimal_actions] = parser.get_episode_aggregate_results()
        rewards_list.append(mean_reward)
        opt_act_list.append(mean_optimal_actions)
    return rewards_list, opt_act_list


def plot_results(expt, rewards_list, opt_act_list, logger):
    logger.info("Plotting the results")
    _, _, episode_count, max_steps, _, labels = expt
    plotter = Plotter(episode_count, max_steps)
    plotter.plot_episode_aggregate_results(rewards_list, opt_act_list, labels)


def get_expt1():
    env = gym.make('MultiArmBandit-v0')
    epsilons = [0.0, 0.01, 0.1]
    agents = [EpsilonGreedyAgent(env.action_space, epsilon=epsilon) for epsilon in epsilons]
    episode_count = 2000
    max_steps = 1000
    outdirs = ['/tmp/expt1_' + str(i) for i in range(len(epsilons))]
    for outdir in outdirs:
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir)
    labels = [r'$\epsilon=0$', r'$\epsilon=0.01$', r'$\epsilon=0.1$']
    return [env, agents, episode_count, max_steps, outdirs, labels]


def get_expt2():
    env = gym.make('MultiArmBandit-v0')
    agent1 = EpsilonGreedyAgent(env.action_space, epsilon=0.0, init_value=5, alpha=0.1, recency_weighting=True)
    agent2 = EpsilonGreedyAgent(env.action_space, epsilon=0.1, alpha=0.1, recency_weighting=True)
    agents = [agent1, agent2]
    episode_count = 2000
    max_steps = 1000
    outdirs = ['/tmp/expt2_' + str(i) for i in range(2)]
    for outdir in outdirs:
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir)
    labels = [r'$Q_0=5, \epsilon=0$', r'$Q_0=0, \epsilon=0.1$']
    return [env, agents, episode_count, max_steps, outdirs, labels]


def get_expt3():
    env = gym.make('MultiArmBandit-v0')
    env.stationary = False
    env.random_walk_var = 0.2

    agent1 = EpsilonGreedyAgent(env.action_space, epsilon=0.1)
    agent2 = EpsilonGreedyAgent(env.action_space, epsilon=0.1, alpha=0.5, recency_weighting=True)
    agents = [agent1, agent2]

    episode_count = 1000
    max_steps = 2000
    outdirs = ['/tmp/expt3_' + str(i) for i in range(2)]
    for outdir in outdirs:
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir)
    labels = ['sample average', 'recency weighting']
    return [env, agents, episode_count, max_steps, outdirs, labels]


def get_expt4():
    env = gym.make('MultiArmBandit-v0')
    agent1 = EpsilonGreedyAgent(env.action_space, epsilon=0.1, alpha=0.1, recency_weighting=True)
    agent2 = EpsilonGreedyAgent(env.action_space, epsilon=0.0, init_value=5, alpha=0.1, recency_weighting=True)
    agent3 = EpsilonGreedyAgent(env.action_space, epsilon=0.0, init_value=10, alpha=0.1, recency_weighting=True)
    agent4 = EpsilonGreedyAgent(env.action_space, epsilon=0.0, init_value=20, alpha=0.1, recency_weighting=True)
    agents = [agent1, agent2, agent3, agent4]
    episode_count = 500
    max_steps = 1000
    outdirs = ['/tmp/expt4_' + str(i) for i in range(4)]
    for outdir in outdirs:
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir)
    labels = [r'$Q_0=0, \epsilon=0.1$',
              r'$Q_0=5, \epsilon=0$',
              r'$Q_0=10, \epsilon=0$',
              r'$Q_0=20, \epsilon=0$']
    return [env, agents, episode_count, max_steps, outdirs, labels]


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    expt = get_expt4()

    run_expt(expt, logger)
    rewards_list, opt_act_list = parse_results(expt, logger)
    plot_results(expt, rewards_list, opt_act_list, logger)
