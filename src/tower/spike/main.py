import random
from collections import Counter
from logging import basicConfig, INFO
from pathlib import Path
from typing import List

import cv2
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from obstacle_tower_env import ObstacleTowerEnv

from tower.spike.action_cycle_check import ActionCycleCheck
from tower.spike.const import Action
from tower.spike.judge_nop import JudgeMove
from tower.spike.jump_cycle_check import JumpCycleCheck
from tower.spike.moving_check_agent import MovingCheckAgent, CheckResult
from tower.spike.util import average_image, frame_pixel_diff

PRJ_ROOT = Path(__file__).parents[3]


def main():
    basicConfig(level=INFO)
    env = ObstacleTowerEnv(str(PRJ_ROOT / 'obstacletower'), retro=False, worker_id=9)
    done = False
    env.floor(1)
    env.reset()

    screen = Screen()
    random_action = RandomRepeatAction(Action.NOP, 0.95)
    judger = JudgeMove()
    last_small_frame = None

    frame_history = FrameHistory(env)
    event_handlers: List[EventHandler] = [
        frame_history,
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

        for h in event_handlers:
            h.after_step()

        didnt_move = judger.did_move(frame_history.last_small_frame, frame_history.current_small_frame, action)

        for h in event_handlers:
            h.end_loop()





class EventHandler:
    def begin_loop(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    def end_loop(self):
        pass


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

    def after_step(self):
        self.current_frame = self.env.render()
        self.current_small_frame = average_image(self.current_frame)
        self.small_frame_pixel_diffs.append(frame_pixel_diff(self.last_small_frame, self.current_small_frame))
        self.small_frame_pixel_diffs = self.small_frame_pixel_diffs[-2:]

    def end_loop(self):
        self.last_frame = self.current_frame
        self.last_small_frame = self.current_small_frame
        self.current_frame = self.last_small_frame = None


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
        else:
            Action.jump_off(self.action)

        return self.action


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
