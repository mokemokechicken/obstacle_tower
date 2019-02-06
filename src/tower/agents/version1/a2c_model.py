from tower.config import Config


class A2CModel:
    def __init__(self, config: Config):
        self.config = config

    def can_load(self):
        return False

    def load_model(self):
        raise NotImplemented()

    def build(self):
        pass


