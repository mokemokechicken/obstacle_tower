from tower.config import Config
from tower.lib.memory import Memory
from tower.observation.event_handlers.base import EventHandler, EventParamsAfterStep


class TrainingDataRecorder(EventHandler):
    def __init__(self, config: Config, memory: Memory):
        self.config = config
        self.memory = memory

    def reset(self):
        pass

    def begin_episode(self, ep: int):
        pass

    def end_episode(self, ep: int):
        pass

    def after_step(self, params: EventParamsAfterStep):
        pass
