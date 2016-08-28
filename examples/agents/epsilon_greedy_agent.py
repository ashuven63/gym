import random
import logging
import gym
import numpy as np


class EpsilonGreedyAgent(object):
    """Expects the driver to call update after every call to act"""
    def __init__(self, action_space, epsilon=0.1, alpha=0.1, recency_weighting=True, init_value=0):
        self.action_space = action_space
        self.epsilon = epsilon
        self.alpha = alpha
        self.recency_weighting = recency_weighting
        # Initialize initial Q estimates and value counts for actions.
        self.prev_action = None
        self.Q = {action: init_value for action in range(self.action_space.n)}
        self.N = {action: 0 for action in range(self.action_space.n)}

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


if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    env = gym.make('MultiArmBandit-v0')

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/epsilon-greedy-agent-multiarmbandit-results'
    env.monitor.start(outdir, force=True, seed=0)
    print env.printEnv()
    # This declaration must go *after* the monitor call, since the
    # monitor's seeding creates a new action_space instance with the
    # appropriate pseudorandom number generator.
    agent = EpsilonGreedyAgent(env.action_space)

    episode_count = 100
    max_steps = 200
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()

        for j in range(max_steps):
            action = agent.act(ob)
            ob, reward, done, _ = env.step(action)
            agent.update(reward)
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Dump result info to disk
    env.monitor.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran EpsilonGreedyAgent")
