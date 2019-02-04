from obstacle_tower_env import ObstacleTowerEnv

from tower.event_handlers.base import EventHandler, EventParamsAfterStep
from tower.spike.util import average_image, frame_pixel_diff


class FrameHistory(EventHandler):
    def __init__(self, env: ObstacleTowerEnv):
        self.env = env
        self.last_frame = None
        self.last_small_frame = None
        self.current_frame = None
        self.current_small_frame = None
        self.small_frame_pixel_diffs = []

    def begin_loop(self):
        if self.last_frame is None:
            self.last_frame = self.env.render()
            self.last_small_frame = average_image(self.last_frame)

    def after_step(self, params: EventParamsAfterStep):
        self.current_frame = self.env.render()
        self.current_small_frame = average_image(self.current_frame)
        self.small_frame_pixel_diffs.append(frame_pixel_diff(self.last_small_frame, self.current_small_frame))
        self.small_frame_pixel_diffs = self.small_frame_pixel_diffs[-2:]

    def end_loop(self):
        self.last_frame = self.current_frame
        self.last_small_frame = self.current_small_frame
        self.current_frame = self.current_small_frame = None
