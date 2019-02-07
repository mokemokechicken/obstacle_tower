import time
from logging import getLogger

import numpy as np

from tower.config import Config
from tower.const import Action
from tower.lib.memory import Memory
from tower.observation.event_handlers.base import EventHandler, EventParamsAfterStep
from tower.observation.manager import ObservationManager

logger = getLogger(__name__)


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
        self.step_info = None

    def begin_episode(self, ep: int):
        self.reset()

    def end_episode(self, ep: int):
        episode_data = {
            "meta": self.create_meta_info(),
            "steps": self.data,
        }
        self.memory.store(episode_data)

    def before_step(self):
        if self.last_obs is None:
            self.last_obs = [0, 0, -1]

        obs = list(self.last_obs)
        obs[0] = (self.observation.frame_history.last_half_frame * 255).astype(np.uint8)
        self.step_info = dict(
            state=obs,
            map=(self.observation.map_observation.image() * 255).astype(np.uint8),
        )

    def after_step(self, params: EventParamsAfterStep):
        self.last_obs = params.obs
        self.step_info['action'] = Action.to_int(params.action)
        self.step_info['reward'] = float(params.reward)
        self.step_info['done'] = params.done
        self.step_info['map_reward'] = self.observation.map_observation.map_reward
        self.data.append(self.step_info)

    def create_meta_info(self):
        meta = dict(
            timestamp=int(time.time()),
            step_len=len(self.data),
            reward=sum([x["reward"] for x in self.data]),
            map_reward=sum([x["map_reward"] for x in self.data]),
        )
        logger.info(str(meta))
        return meta

