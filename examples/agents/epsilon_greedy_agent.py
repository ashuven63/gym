import random
import numpy as np


class EpsilonGreedyAgent(object):
    """Expects the driver to call update after every call to act"""
    def __init__(self, action_space, epsilon=0.1, alpha=0.1, recency_weighting=True):
        self.action_space = action_space
        self.epsilon = epsilon
        self.alpha = alpha
        self.recency_weighting = recency_weighting
        # Initialize initial Q estimates and value counts for actions.
        self.prev_action = None
        self.Q = {action: 0 for action in range(0, self.action_space.N)}
        self.N = {action: 0 for action in range(0, self.action_space.N)}

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
        best_value = max([self.Q[action] for action in range(0, self.action_space.N)])
        optimal_actions = [action for action in self.Q if self.Q[action] == best_value]
        assert (len(optimal_actions) > 0)
        # Return randomly from optimal actions.
        return random.choice(optimal_actions)

    def update(self, reward):
        """Update action estimates after receiving reward from environment"""
        self.N[self.prev_action] += 1
        if self.recency_weighting:
            self.Q[self.prev_action] += ((reward - self.Q[self.prev_action]) / self.N[self.prev_action])
        else:
            self.Q[self.prev_action] += self.alpha * (reward - self.Q[self.prev_action])