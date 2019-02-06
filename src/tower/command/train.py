from logging import getLogger

from tower.agents.version1.state_model import StateModel
from tower.config import Config

logger = getLogger(__name__)


def start(config: Config):
    TrainCommand(config).start()


class TrainCommand:
    def __init__(self, config: Config):
        self.config = config

    def start(self):
        state_model = StateModel(self.config)
        state_model.build()
