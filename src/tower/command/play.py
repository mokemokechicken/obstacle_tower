from logging import getLogger
from typing import List
import numpy as np
import cv2
from obstacle_tower_env import ObstacleTowerEnv

from tower.actors.random_repeat_actor import RandomRepeatActor
from tower.config import Config
from tower.const import Action
from tower.event_handlers.base import EventHandler, EventParamsAfterStep
from tower.event_handlers.frame import FrameHistory
from tower.event_handlers.map_observation import MapObservation
from tower.event_handlers.moving_checker import MovingChecker
from tower.event_handlers.position_estimator import PositionEstimator
from tower.lib.screen import Screen

logger = getLogger(__name__)


def start(config: Config):
    PlayCommand(config).start()


class PlayCommand:
    def __init__(self, config: Config):
        self.config = config

    def start(self):
        env = ObstacleTowerEnv(str(self.config.resource.obstacle_tower_path), retro=False, worker_id=9)
        done = False
        env.floor(1)
        env.reset()

        screen = Screen(render=self.config.play.render)
        random_actor = RandomRepeatActor(continue_rate=0.9)
        random_actor.reset(schedules=[
            (Action.CAMERA_RIGHT, 3),
            (Action.CAMERA_LEFT, 6),
            (Action.CAMERA_RIGHT, 3),
            (Action.NOP, 5),
            (Action.FORWARD, 8),
            (Action.RIGHT, 2),
            (Action.LEFT, 4),
            (Action.RIGHT, 2),
        ])

        frame_history = FrameHistory(env)
        moving_checker = MovingChecker(frame_history)
        position_estimator = PositionEstimator(moving_checker)
        map_observation = MapObservation(position_estimator, moving_checker)
        event_handlers: List[EventHandler] = [
            frame_history,
            moving_checker,
            position_estimator,
            map_observation,
        ]

        while not done:
            for h in event_handlers:
                h.begin_loop()

            screen.show("original", frame_history.last_frame)
            cv2.waitKey(self.config.play.wait_per_frame)

            for h in event_handlers:
                h.before_step()

            action = random_actor.decide_action(moving_checker.did_move)
            obs, reward, done, info = env.step(action)
            if reward != 0:
                logger.info(f"Get Reward={reward} Keys={obs[1]}")
            # logger.info(f"Keys={obs[1]} Time Remain={obs[2]}")

            params = EventParamsAfterStep(action, obs, reward, done, info)
            for h in event_handlers:
                h.after_step(params)

            screen.show("map", map_observation.concat_images())

            if len(frame_history.small_frame_pixel_diffs) > 0:
                f1 = frame_history.small_frame_pixel_diffs[-1]
                if len(frame_history.small_frame_pixel_diffs) > 1:
                    f2 = frame_history.small_frame_pixel_diffs[-2]
                    f1 = np.concatenate((f2, f1), axis=1)
                screen.show("diff", f1)

            for h in event_handlers:
                h.end_loop()
