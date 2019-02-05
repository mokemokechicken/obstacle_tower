from typing import List

from obstacle_tower_env import ObstacleTowerEnv

from tower.config import Config
from tower.observation.base import EventHandler, EventParamsAfterStep
from tower.observation.frame import FrameHistory
from tower.observation.map_observation import MapObservation
from tower.observation.moving_checker import MovingChecker
from tower.observation.position_estimator import PositionEstimator


class ObservationManager:
    def __init__(self, config: Config, env: ObstacleTowerEnv):
        self.config = config
        self._env = env
        self.frame_history: FrameHistory = None
        self.moving_checker: MovingChecker = None
        self.position_estimator: PositionEstimator = None
        self.map_observation: MapObservation = None
        self._event_handlers: List[EventHandler] = []

    def step(self, action):
        self.before_step()
        obs, reward, done, info = self._env.step(action)
        params = EventParamsAfterStep(action, obs, reward, done, info)
        self.after_step(params)
        return obs, reward, done, info

    def reset(self):
        ret = self._env.reset()

        self.frame_history = FrameHistory(self._env)
        self.moving_checker = MovingChecker(self.frame_history)
        self.position_estimator = PositionEstimator(self.moving_checker)
        self.map_observation = MapObservation(self.position_estimator, self.moving_checker)
        self._event_handlers = [
            self.frame_history,
            self.moving_checker,
            self.position_estimator,
            self.map_observation,
        ]
        return ret

    def render(self):
        return self._env.render()

    def floor(self, floor):
        return self._env.floor(floor)

    def begin_loop(self):
        for h in self._event_handlers:
            h.begin_loop()

    def before_step(self):
        for h in self._event_handlers:
            h.before_step()

    def after_step(self, params):
        for h in self._event_handlers:
            h.after_step(params)

    def end_loop(self):
        for h in self._event_handlers:
            h.end_loop()
