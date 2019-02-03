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
from tower.spike.jump_cycle_check import JumpCycleCheck
from tower.spike.moving_check_agent import MovingCheckAgent, CheckResult

PRJ_ROOT = Path(__file__).parents[3]


def main():
    basicConfig(level=INFO)
    env = ObstacleTowerEnv(str(PRJ_ROOT / 'obstacletower'), retro=False, worker_id=9)
    done = False
    frames = []
    env.floor(1)
    env.reset()

    screen = Screen()
    mc_agent = MovingCheckAgent(env)
    jc_agent = ActionCycleCheck(env, Action.JUMP)

    for _ in range(20):
        env.step(Action.FORWARD)

    while not done:
        frame = env.render()
        frames.append(frame)
        screen.show("original", frame)

        # small_image = average_image(frame)
        # screen.show("mean", small_image)
        # if last_av_image is not None:
        #     diff_av_image = np.mean(np.abs(avg_image - last_av_image), axis=2)
        #     screen.show("diff", diff_av_image)
        # last_av_image = avg_image
        cv2.waitKey(1)
        # obs, reward, done, info = env.step(env.action_space.sample())

        # obs, reward, done, info = env.step(cc.action)
        # cc.prev_small_image(small_image)

        # jc_agent.step()
        # if jc_agent.done:
        #     print(jc_agent.estimated_cycle)
        #     break

        mc_agent.step()
        if mc_agent.done:
            mc_agent.update_model()
            mc_agent.reset()


ROTATION_CYCLE = 20


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
