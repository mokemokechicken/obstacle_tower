from collections import OrderedDict

from obstacle_tower_env import ObstacleTowerEnv

from tower.config import Config
from tower.observation.event_handlers.base import EventHandler, EventParamsAfterStep
from tower.observation.event_handlers.frame import FrameHistory
from tower.observation.event_handlers.map_observation import MapObservation
from tower.observation.event_handlers.moving_checker import MovingChecker
from tower.observation.event_handlers.position_estimator import PositionEstimator


class ObservationManager:
    def __init__(self, config: Config, env: ObstacleTowerEnv):
        self.config = config
        self._env = env
        self._event_handlers: OrderedDict[str, EventHandler] = OrderedDict()

    def setup(self):
        frame_history = FrameHistory(self._env)
        moving_checker = MovingChecker(frame_history)
        position_estimator = PositionEstimator(moving_checker)
        map_observation = MapObservation(self.config, position_estimator, moving_checker)

        self.add_event_handler("frame_history", frame_history)
        self.add_event_handler("moving_checker", moving_checker)
        self.add_event_handler("position_estimator", position_estimator)
        self.add_event_handler("map_observation", map_observation)

    def step(self, action):
        self.before_step()
        obs, reward, done, info = self._env.step(action)
        params = EventParamsAfterStep(action, obs, reward, done, info)
        self.after_step(params)
        return obs, reward, done, info

    def reset(self):
        ret = self._env.reset()
        for h in self._event_handlers.values():
            h.reset()
        return ret

    def add_event_handler(self, name: str, h: EventHandler):
        assert isinstance(h, EventHandler)
        self._event_handlers[name] = h

    def get_event_handler(self, name: str):
        return self._event_handlers.get(name)

    @property
    def frame_history(self):
        return self.get_event_handler("frame_history")

    @property
    def moving_checker(self):
        return self.get_event_handler("moving_checker")

    @property
    def position_estimator(self):
        return self.get_event_handler("position_estimator")

    @property
    def map_observation(self):
        return self.get_event_handler("map_observation")

    def render(self):
        return self._env.render()

    def floor(self, floor):
        return self._env.floor(floor)

    def begin_loop(self):
        for h in self._event_handlers.values():
            h.begin_loop()

    def before_step(self):
        for h in self._event_handlers.values():
            h.before_step()

    def after_step(self, params):
        for h in self._event_handlers.values():
            h.after_step(params)

    def end_loop(self):
        for h in self._event_handlers.values():
            h.end_loop()
