import gym

from tower.config import Config


class AgentBase:
    def __init__(self, config: Config, env: gym.Env):
        self.config = config
        self.env = env

    def setup(self):
        pass

    def play(self):
        pass


class TrainerBase:
    def __init__(self, config: Config):
        self.config = config

    def setup(self):
        pass

    def train(self):
        pass
