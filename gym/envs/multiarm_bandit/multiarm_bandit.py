import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class MultiArmBandit(gym.Env):

    def __init__(self,
                 num_arms=10,
                 value_mean=0,
                 value_var=1,
                 reward_var=1,
                 stationary=True,
                 random_walk_var=0.01):
        self.value_mean = value_mean
        self.value_var = value_var
        self.reward_var = reward_var
        self.stationary = stationary
        self.action_space = spaces.Discrete(num_arms)
        self._seed()
        self._reset()

    def init_q_star(self):
        # TODO(abora) : figure out how to use random seed
        q_star = {}
        for action in range(0, self.action_space.n):
            q_star[action] = np.random.normal(loc=self.value_mean, scale=self.value_var)
        return q_star

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        reward = np.random.normal(loc=self.q_star[action], scale=self.reward_var)
        if not self.stationary:
            self.random_walk()
        return (None, reward, False, {})

    def random_walk(self):
        for action in range(0, self.action_space.n):
            self.q_star[action] += np.random.normal(loc=0.0, scale=self.random_walk_var)

    def _reset(self):
        self.q_star = self.init_q_star()
