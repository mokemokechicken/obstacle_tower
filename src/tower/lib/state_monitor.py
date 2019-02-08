import cv2

from tower.agents.version1.state_model import StateModel
from tower.observation.event_handlers.base import EventHandler
from tower.observation.event_handlers.frame import FrameHistory
from tower.observation.event_handlers.infomation import InformationHandler

import numpy as np


class StateMonitor(EventHandler):
    def __init__(self, state_model: StateModel, frame_history: FrameHistory, info: InformationHandler):
        self.state_model = state_model
        self.frame_history = frame_history
        self.info = info
        self.plots = None

    def before_step(self):
        state, sigma = self.state_model.encode_to_state(self.frame_history.last_half_frame)
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
        self.info.screen.show("state", image)
