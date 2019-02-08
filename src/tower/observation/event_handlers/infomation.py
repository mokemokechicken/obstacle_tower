from tower.config import Config
from tower.lib.screen import Screen
from tower.observation.event_handlers.base import EventHandler
from tower.observation.manager import ObservationManager

import cv2
import numpy as np


class InformationHandler(EventHandler):
    def __init__(self, config: Config, observation: ObservationManager):
        self.config = config
        self.screen = Screen(render=True)
        self.obs = observation

    def begin_loop(self):
        obs = self.obs
        self.screen.show("original", obs.frame_history.last_frame)
        self.screen.show("map", obs.map_observation.concat_images())

        if len(obs.frame_history.small_frame_pixel_diffs) > 0:
            f1 = obs.frame_history.small_frame_pixel_diffs[-1]
            if len(obs.frame_history.small_frame_pixel_diffs) > 1:
                f2 = obs.frame_history.small_frame_pixel_diffs[-2]
                f1 = np.concatenate((f2, f1), axis=1)
            self.screen.show("diff", f1)

    def end_loop(self):
        cv2.waitKey(self.config.play.wait_per_frame)
