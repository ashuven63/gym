import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class MultiArmedBandit(gym.Env):

    def __init__(self, num_arms=10, value_mean=0, value_var=1, reward_var=1):
        self.value_mean = value_mean
        self.value_var = value_var
        self.reward_var = reward_var
        self.action_space = spaces.Discrete(num_arms)
        self._seed()
        self._reset()

    def init_q_star(self):
        # TODO(abora) : figure out how to use random seed
        q_star = {}
        for action in self.action_space:
            q_star[action] = np.random.normal(loc=self.value_mean, scale=self.value_var)
        return q_star

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.q_star = self.init_q_star()
