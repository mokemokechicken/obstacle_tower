import gym

from tower.config import Config


class AgentBase:
    def __init__(self, config: Config, env: gym.Env, env_id=1):
        self.config = config
        self.env = env
        self.env_id = env_id

    def setup(self):
        pass

    def play(self):
        pass
