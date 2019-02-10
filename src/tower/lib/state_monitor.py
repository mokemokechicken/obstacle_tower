from logging import getLogger

import cv2

from tower.agents.version1.state_model import StateModel
from tower.observation.event_handlers.base import EventHandler
from tower.observation.event_handlers.frame import FrameHistory
from tower.observation.event_handlers.infomation import InformationHandler

import numpy as np

logger = getLogger(__name__)


class StateMonitor(EventHandler):
    def __init__(self, state_model: StateModel, frame_history: FrameHistory, info: InformationHandler):
        self.state_model = state_model
        self.frame_history = frame_history
        self.info = info
        self.plots = None
        self._memory: StateMemory = None
        self.cum_rarity = 0.

    def get_memory(self, state):
        if self._memory is None:
            self._memory = StateMemory(state_size=len(state))
        return self._memory

    def begin_episode(self, ep: int):
        self._memory = None
        self.cum_rarity = 0

    def before_step(self):
        half_frame = self.frame_history.last_half_frame
        state, sigma = self.state_model.encode_to_state(half_frame)
        frame = self.state_model.decode_from_state(state)
        # self.info.screen.show("reconstruct", frame)
        self.plot(frame, state, sigma)

    def plot(self, frame, state, sigma):
        fh, fw, _ = frame.shape
        size = len(state)
        bar_w = 10
        bar_height_per_val = fw // 6
        margin_w = 1
        pw = (bar_w+margin_w*2)
        w, h = size * pw, fw

        state_img = np.zeros((h, w, 3), dtype=np.uint8)
        for i, (val, sig) in enumerate(zip(state, sigma)):
            height = int(np.clip(val * bar_height_per_val, -h/2, h/2))
            sig = int(sig * bar_height_per_val)
            cv2.rectangle(state_img, (i*pw + margin_w, h//2-height), (i*pw+margin_w+bar_w, h//2), (255, 0, 0), -1)
            cv2.rectangle(state_img, (i*pw + margin_w+bar_w//3, h//2-height-sig), (i*pw+margin_w+2*bar_w//3,
                                                                                   h//2-height+sig),
                          (0, 0, 255), -1)

        image = np.zeros((max(fh, h), fw+w+margin_w, 3), dtype=np.float)  # (h, w, ch)
        image[0:fh, 0:fw, :] = frame
        image[0:h, fw+margin_w:fw+margin_w+w, :] = state_img / 255

        # memory
        memory = self.get_memory(state)
        differences = memory.difference_array(state)
        rarity = float(0. if differences is None else np.min(differences))
        self.cum_rarity = self.cum_rarity * 0.95 + rarity
        # logger.info(f"rarity={rarity:.2f}")
        memory.store(state)
        cv2.rectangle(image, (fw, 0), (fw+min(w, int(self.cum_rarity * 100)), 20), (0, 255, 0), thickness=-1)
        cv2.rectangle(image, (fw, image.shape[0] - min(30, int(rarity*100))), (fw+w, image.shape[0]),
                      (0, 0, 255), thickness=-1)
        self.info.screen.show("state", image)


class StateMemory:
    def __init__(self, state_size, memory_size=1000):
        self.memory_size = memory_size
        self.state_size = state_size
        self.index = 0
        self._memory = np.zeros((memory_size, state_size), dtype=np.float32)

    @property
    def memory(self):
        return self._memory

    def store(self, state):
        index = self.index % self.memory_size
        self.memory[index, ] = state
        self.index += 1

    def difference_array(self, state):
        max_index = min(self.memory_size, self.index)
        if max_index > 0:
            return np.sum(np.square(self.memory[:max_index, ] - state), axis=1)
        else:
            return None
