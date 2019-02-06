import time

from tower.config import Config
from tower.lib.memory import Memory
from tower.observation.event_handlers.base import EventHandler, EventParamsAfterStep
from tower.observation.manager import ObservationManager
from tower.spike.util import average_image
import numpy as np


class TrainingDataRecorder(EventHandler):
    def __init__(self, config: Config, memory: Memory, obs: ObservationManager):
        self.config = config
        self.memory = memory
        self.observation = obs
        self.data = []
        self.step_info: dict = None
        self.last_obs = None

        self.memory.setup()

    def reset(self):
        self.data = []
        self.last_obs = None

    def begin_episode(self, ep: int):
        self.reset()

    def end_episode(self, ep: int):
        data = {
            "meta": self.create_meta_info(),
            "data": self.data,
        }
        self.memory.store(data)

    def before_step(self):
        if self.last_obs is None:
            self.step_info = None
            return

        obs = list(self.last_obs)
        obs[0] = (self.observation.frame_history.last_half_frame * 255).astype(np.uint8)
        self.step_info = dict(
            state=obs,
            map=(self.observation.map_observation.image() * 255).astype(np.uint8),
        )

    def after_step(self, params: EventParamsAfterStep):
        self.last_obs = params.obs
        if self.step_info is None:
            return

        self.step_info['action'] = params.action
        self.step_info['reward'] = params.reward
        self.step_info['done'] = params.done
        self.step_info['map_reward'] = 1 - self.observation.map_observation.last_visit_value
        self.data.append(self.step_info)

    def create_meta_info(self):
        meta = dict(
            timestamp=int(time.time()),
            step_len=len(self.data),
            reward=sum([x["reward"] for x in self.data]),
            map_reward=sum([x["map_reward"] for x in self.data]),
        )
        return meta
