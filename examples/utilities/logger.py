import pickle


class Logger(object):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.log = []

    def update_log(self, iteration):
        env_state = self.env.get_state()
        agent_state = self.agent.get_state()
        self.log.append((env_state, agent_state))

    def dump_log(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.log, f)
