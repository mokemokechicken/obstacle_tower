from logging import basicConfig, INFO
from pathlib import Path
from typing import List

import cv2
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from obstacle_tower_env import ObstacleTowerEnv

from tower.event_handlers.base import EventParamsAfterStep, EventHandler, FrameHistory, PositionEstimator
from tower.const import Action
from tower.event_handlers.judge_move import JudgeMove

PRJ_ROOT = Path(__file__).parents[3]


def main():
    basicConfig(level=INFO)
    env = ObstacleTowerEnv(str(PRJ_ROOT / 'obstacletower'), retro=False, worker_id=9)
    done = False
    env.floor(1)
    env.reset()

    screen = Screen()
    random_action = RandomRepeatAction(Action.NOP, 0.95)

    frame_history = FrameHistory(env)
    judger = JudgeMove(frame_history)
    position_estimator = PositionEstimator()
    event_handlers: List[EventHandler] = [
        frame_history,
        judger,

    ]

    while not done:
        for h in event_handlers:
            h.begin_loop()

        screen.show("original", frame_history.last_frame)
        cv2.waitKey(0)

        for h in event_handlers:
            h.before_step()

        action = random_action.decide_action()
        obs, reward, done, info = env.step(action)

        params = EventParamsAfterStep(action, obs, reward, done, info)
        for h in event_handlers:
            h.after_step(params)

        judger.did_move(frame_history.last_small_frame, frame_history.current_small_frame, action)

        if len(frame_history.small_frame_pixel_diffs) > 0:
            screen.show("diff0", frame_history.small_frame_pixel_diffs[-1])

        if len(frame_history.small_frame_pixel_diffs) > 1:
            screen.show("diff1", frame_history.small_frame_pixel_diffs[-2])

        for h in event_handlers:
            h.end_loop()


class RandomRepeatAction:
    action = None
    continue_rate = None

    def __init__(self, action=None, continue_rate=0.95):
        self.reset(action, continue_rate)

    def reset(self, action=None, continue_rate=None):
        if continue_rate is not None:
            self.continue_rate = continue_rate
        self.action = action

    def decide_action(self):
        if self.action is None or np.random.random() >= self.continue_rate:
            self.action = Action.sample_action()
            self.to_move_or_camera(self.action)
        return self.action

    @staticmethod
    def to_move_or_camera(action):
        if action[Action.IDX_MOVE_FB] + action[Action.IDX_MOVE_RL] > 0 and action[Action.IDX_CAMERA_LR] > 0:
            if np.random.random() < 0.5:
                action[Action.IDX_CAMERA_LR] = 0
            else:
                action[Action.IDX_MOVE_FB] = 0
                action[Action.IDX_MOVE_RL] = 0


class Screen:
    def __init__(self):
        self._windows = {}

    def show(self, window_name, image):
        if window_name not in self._windows:
            self.setup_window(window_name)
        cv2.imshow(window_name, image)

    def setup_window(self, window_name):
        idx = len(self._windows)
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, (idx % 3) * 400, (idx // 3) * 300)
        self._windows[window_name] = 1


def output_to_movie(frames):
    fig = plt.gcf()
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
    anim.save("tower.mp4", writer='ffmpeg')
