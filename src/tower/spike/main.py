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
from tower.spike.util import average_image

PRJ_ROOT = Path(__file__).parents[3]


def main():
    basicConfig(level=INFO)
    env = ObstacleTowerEnv(str(PRJ_ROOT / 'obstacletower'), retro=False, worker_id=9)
    done = False
    env.floor(1)
    env.reset()

    screen = Screen()
    random_action = RandomRepeatAction(Action.NOP, 10)
    judger = JudgeMove()

    for _ in range(10):
        env.step(Action.FORWARD)

    last_small_frame = None

    while not done:
        frame = env.render()

        if last_small_frame is None:
            last_small_frame = average_image(frame)

        action = random_action.decide_action()
        if action is None:
            random_action.reset()
            action = random_action.decide_action()

        env.step(action)
        current_small_frame = average_image(env.render())
        didnt_move = judger.did_move(last_small_frame, current_small_frame, action)
        last_small_frame = current_small_frame

        screen.show("original", frame)
        cv2.waitKey(0)


ROTATION_CYCLE = 20


class RandomRepeatAction:
    action = None
    n = None

    def __init__(self, action=None, n=None):
        self.reset(action, n)

    def reset(self, action=None, n=None):
        if action is None:
            actions = [Action.NOP, Action.FORWARD, Action.BACK,
                       Action.CAMERA_RIGHT, Action.CAMERA_LEFT,
                       Action.JUMP,
                       Action.LEFT, Action.RIGHT,
                       ]
            action = actions[np.random.choice(range(len(actions)))]
        self.action = action
        self.n = n or int(np.random.randint(4, 10))

    def decide_action(self):
        if self.n == 0:
            return None
        self.n -= 1
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
