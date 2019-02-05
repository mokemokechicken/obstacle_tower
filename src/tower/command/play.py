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
from tower.observation.manager import ObservationManager

logger = getLogger(__name__)


def start(config: Config):
    PlayCommand(config).start()


class PlayCommand:
    def __init__(self, config: Config):
        self.config = config
        self.screen: Screen = None

    def start(self):
        env = ObstacleTowerEnv(str(self.config.resource.obstacle_tower_path), retro=False, worker_id=9)
        self.screen = Screen(render=self.config.play.render)

        obs = ObservationManager(self.config, env)
        done = False
        obs.floor(1)
        obs.reset()
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

        while not done:
            obs.begin_loop()
            self.show_information(obs)

            action = random_actor.decide_action(obs.moving_checker.did_move)
            obs, reward, done, info = env.step(action)
            if reward != 0:
                logger.info(f"Get Reward={reward} Keys={obs[1]}")
            obs.end_loop()

    def show_information(self, obs: ObservationManager):
        self.screen.show("original", obs.frame_history.last_frame)
        self.screen.show("map", obs.map_observation.concat_images())
        cv2.waitKey(self.config.play.wait_per_frame)

        if len(obs.frame_history.small_frame_pixel_diffs) > 0:
            f1 = obs.frame_history.small_frame_pixel_diffs[-1]
            if len(obs.frame_history.small_frame_pixel_diffs) > 1:
                f2 = obs.frame_history.small_frame_pixel_diffs[-2]
                f1 = np.concatenate((f2, f1), axis=1)
            self.screen.show("diff", f1)
