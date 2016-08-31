import random
import logging
import sys
import gym
import os
import numpy as np

sys.path.append("../utilities")
from logger import Logger
from parser import Parser
from plotter import Plotter


class EpsilonGreedyAgent(object):
    """Expects the driver to call update after every call to act"""
    def __init__(self, action_space, epsilon=0.0, alpha=0.1, recency_weighting=False, init_value=0):
        self.action_space = action_space
        self.epsilon = epsilon
        self.alpha = alpha
        self.recency_weighting = recency_weighting
        self.init_value = init_value
        # Initialize initial Q estimates and value counts for actions.
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
        self.Q = {action: self.init_value for action in range(self.action_space.n)}
        self.N = {action: 0 for action in range(self.action_space.n)}

    def __str__(self):
        print 'Action\tEstimate\tFrequency'
        for action in self.Q:
            print '{0}\t{1}\t{2}'.format(action, self.Q[action], self.N[action])
        print 'Action{0}'.format(self.prev_action)
        print 'Reward{0}'.format(self.cur_reward)
        return ""

if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    env = gym.make('MultiArmBandit-v0')
    episode_count = 20
    max_steps = 100

    epsilons = [0.0, 0.01, 0.1]
    agents = [EpsilonGreedyAgent(env.action_space, epsilon=epsilon) for epsilon in epsilons]
    outdirs = ['/tmp/epsilon-greedy-agent-multiarmbandit-results' + str(i) for i in range(len(epsilons))]

    for outdir in outdirs:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    # agent = EpsilonGreedyAgent(env.action_space, epsilon=0.0, init_value=5, alpha=0.1, recency_weighting=True)
    # outdir = '/tmp/epsilon-greedy-agent-multiarmbandit-results'

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

    logger.info("Parsing the results")
    rewards_list, opt_act_list = [], []
    for outdir in outdirs:
        parser = Parser(outdir, episode_count, max_steps)
        [mean_reward, mean_optimal_actions] = parser.get_episode_aggregate_results()
        rewards_list.append(mean_reward)
        opt_act_list.append(mean_optimal_actions)

    logger.info("Plotting the results")
    plotter = Plotter(episode_count, max_steps)
    plotter.plot_episode_aggregate_results(rewards_list, opt_act_list)
