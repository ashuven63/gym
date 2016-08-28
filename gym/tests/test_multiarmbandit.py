import logging
import os, sys
from gym.examples.agents.random_agent import RandomAgent
import gym

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
outdir = '/tmp/random-agent-multiarmbandit-results'
env.monitor.start(outdir, force=True, seed=0)

# This declaration must go *after* the monitor call, since the
# monitor's seeding creates a new action_space instance with the
# appropriate pseudorandom number generator.
agent = RandomAgent(env.action_space)

episode_count = 100
max_steps = 200
reward = 0
done = False

for i in range(episode_count):
    ob = env.reset()

    for j in range(max_steps):
        action = agent.act(None, reward, done)
        ob, reward, done, _ = env.step(action)
        if done:
            break
        # Note there's no env.render() here. But the environment still can open window and
        # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
        # Video is not recorded every episode, see capped_cubic_video_schedule for details.

# Dump result info to disk
env.monitor.close()

# Upload to the scoreboard. We could also do this from another
# process if we wanted.
logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
gym.upload(outdir)
