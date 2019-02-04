from logging import basicConfig, INFO, getLogger
from pathlib import Path
from typing import List

import cv2
from matplotlib import animation
from matplotlib import pyplot as plt
from obstacle_tower_env import ObstacleTowerEnv

from tower.actors.random_repeat_actor import RandomRepeatActor
from tower.const import Action
from tower.event_handlers.base import EventParamsAfterStep, EventHandler
from tower.event_handlers.frame import FrameHistory
from tower.event_handlers.judge_move import JudgeMove
from tower.event_handlers.position_estimator import PositionEstimator

PRJ_ROOT = Path(__file__).parents[3]


logger = getLogger(__name__)


def main():
    basicConfig(level=INFO)
    env = ObstacleTowerEnv(str(PRJ_ROOT / 'obstacletower'), retro=False, worker_id=9)
    done = False
    env.floor(1)
    env.reset()

    screen = Screen()
    random_actor = RandomRepeatActor(Action.NOP, 0.95)
    random_actor.reset(schedules=[
        (Action.NOP, 5),
        (Action.FORWARD, 10),
        (Action.RIGHT, 5),
        (Action.LEFT, 10),
        (Action.RIGHT, 5),
        (Action.CAMERA_RIGHT, 5),
        (Action.CAMERA_LEFT, 10),
        (Action.CAMERA_RIGHT, 5),
    ])

    frame_history = FrameHistory(env)
    judger = JudgeMove(frame_history)
    position_estimator = PositionEstimator(judger)
    event_handlers: List[EventHandler] = [
        frame_history,
        judger,
        position_estimator,
    ]

    while not done:
        for h in event_handlers:
            h.begin_loop()

        screen.show("original", frame_history.last_frame)
        cv2.waitKey(0)

        for h in event_handlers:
            h.before_step()

        action = random_actor.decide_action()
        obs, reward, done, info = env.step(action)

        params = EventParamsAfterStep(action, obs, reward, done, info)
        for h in event_handlers:
            h.after_step(params)

        if len(frame_history.small_frame_pixel_diffs) > 0:
            screen.show("diff0", frame_history.small_frame_pixel_diffs[-1])

        if len(frame_history.small_frame_pixel_diffs) > 1:
            screen.show("diff1", frame_history.small_frame_pixel_diffs[-2])

        for h in event_handlers:
            h.end_loop()


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
