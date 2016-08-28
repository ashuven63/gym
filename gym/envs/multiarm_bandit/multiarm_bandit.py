import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class MultiArmedBandit(gym.Env):

    def __init__(self, num_arms=10, val_mean=0, val_var=1, reward_var=1):
    	self.action_space = spaces.Discrete(num_arms)
    	self._seed()
        self._reset()

    def init_q_star(self):
    	return np.random.normal(loc=val_mean, scale=val_var, size=[num_arms])

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
