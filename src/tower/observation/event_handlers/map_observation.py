import math
from logging import getLogger

from tower.config import Config
from tower.const import Action
from tower.observation.event_handlers.base import EventHandler, EventParamsAfterStep
from tower.observation.event_handlers.moving_checker import MovingChecker
from tower.observation.event_handlers.position_estimator import PositionEstimator

import numpy as np

logger = getLogger(__name__)


class DirectionMap:
    def __init__(self, size):
        self.size = size
        self._cache = {}

    def get_map(self, direction: int):
        direction = int(direction)
        if direction not in self._cache:
            self._cache[direction] = self._make_map(direction)
        return self._cache[direction]

    def _make_map(self, direction: int):
        rad = (direction * 18) * math.pi / 180
        dx = math.cos(rad)
        dy = math.sin(rad)
        dir_map = np.zeros((self.size, self.size), dtype=np.float32)
        px = py = self.size // 2
        while 0 <= px < self.size and 0 <= py < self.size:
            dir_map[int(py), int(px)] = 1.
            px += dx
            py += dy
        return dir_map


class MapController:
    def __init__(self, size=64, min_value=0., max_value=1., name=None, scale=1.):
        assert size % 2 == 0
        self.size = size
        self.map = np.zeros((size * 3, size * 3), dtype=np.float16)  # (y, x)
        self.offset_origin_x = self.offset_origin_y = int(size * 1.5)
        self.min_value = min_value
        self.max_value = max_value
        self.name = name
        self.scale = scale

    def add_value(self, x: int, y: int, value: float):
        x, y = int(x/self.scale), int(y/self.scale)
        self._check_view_bounding_box(x, y)
        mx = x + self.offset_origin_x
        my = y + self.offset_origin_y
        value = self.map[my, mx] = float(np.clip(self.map[my, mx] + value, self.min_value, self.max_value))
        logger.debug(f"map {self.name or ''}({x},{y})={self.map[my, mx]}")
        return value

    def fetch_around(self, x: int, y: int) -> np.ndarray:
        x, y = int(x/self.scale), int(y/self.scale)
        self._check_view_bounding_box(x, y)
        x0, y0, x1, y1 = self.view_box(x, y)
        return self.map[y0:y1, x0:x1]

    def view_box(self, x, y):
        x0 = x - self.size // 2 + self.offset_origin_x
        y0 = y - self.size // 2 + self.offset_origin_y
        x1 = x + self.size // 2 + self.offset_origin_x
        y1 = y + self.size // 2 + self.offset_origin_y
        return x0, y0, x1, y1

    def _check_view_bounding_box(self, x, y):
        x0, y0, x1, y1 = self.view_box(x, y)
        if x0 < 0 or y0 < 0 or self.map.shape[1] <= x1 or self.map.shape[0] <= y1:
            self._expand_map(x, y)

    def _expand_map(self, x, y):
        x0, y0, x1, y1 = self.view_box(x, y)
        sy, sx = self.map.shape
        if x0 < 0:
            exp_size = math.ceil(- x0 / self.size) * self.size
            new_map = np.zeros((sy, sx + exp_size), dtype=self.map.dtype)
            new_map[:, exp_size:sx + exp_size] = self.map
            del self.map
            self.map = new_map
            self.offset_origin_x += exp_size
        elif sx <= x1:
            exp_size = math.ceil((x1 - sx + 1) / self.size) * self.size
            new_map = np.zeros((sy, sx + exp_size), dtype=self.map.dtype)
            new_map[:, 0:sx] = self.map
            del self.map
            self.map = new_map
            # offset_origin_x is not changed

        sy, sx = self.map.shape
        if y0 < 0:
            exp_size = math.ceil(- y0 / self.size) * self.size
            new_map = np.zeros((sy + exp_size, sx), dtype=self.map.dtype)
            new_map[exp_size:sy + exp_size, :] = self.map
            del self.map
            self.map = new_map
            self.offset_origin_y += exp_size
        elif sy <= y1:
            exp_size = math.ceil((y1 - sy + 1) / self.size) * self.size
            new_map = np.zeros((sy + exp_size, sx), dtype=self.map.dtype)
            new_map[0:sy, :] = self.map
            del self.map
            self.map = new_map
            # offset_origin_y is not changed
        logger.info(f"expanded map to {self.map.shape}")


class MapObservation(EventHandler):
    visit_map: MapController = None
    wall_map: MapController = None
    dir_map: DirectionMap = None

    def __init__(self, config: Config, position_estimator: PositionEstimator, moving_checker: MovingChecker):
        self.config = config
        self.pos_est = position_estimator
        self.moving_checker = moving_checker
        #
        mc = self.config.map
        self.VISIT_SCALE = mc.visit_map_scale
        self.VISIT_VALUE = mc.visit_map_value
        self.WALL_SCALE = mc.wall_map_scale
        self.WALL_VALUE = mc.wall_map_value
        self.size = mc.map_size
        self.last_visit_value = 0
        self.reset()

    def reset(self):
        self.visit_map = MapController(size=self.size, name="visit", scale=self.VISIT_SCALE)
        self.wall_map = MapController(size=self.size, name="wall", scale=self.WALL_SCALE)
        self.dir_map = DirectionMap(size=self.size)
        self.last_visit_value = 0

    @property
    def map_reward(self):
        return 1 - self.last_visit_value

    def after_step(self, params: EventParamsAfterStep):
        self.last_visit_value = self.visit_map.add_value(self.pos_est.px, self.pos_est.py, self.VISIT_VALUE)

        action = params.action
        if not self.moving_checker.did_move and (action[Action.IDX_MOVE_FB] > 0 or action[Action.IDX_MOVE_RL] > 0):
            self.wall_map.add_value(self.pos_est.px + self.pos_est.dx, self.pos_est.py + self.pos_est.dy,
                                    self.WALL_VALUE)

    def image(self, dtype=np.float32):
        visit_image = self.get_visit_map_image()
        wall_image = self.get_wall_map_image()
        dir_image = self.get_direction_image()

        visit_image = np.expand_dims(visit_image, axis=2)
        wall_image = np.expand_dims(wall_image, axis=2)
        dir_image = np.expand_dims(dir_image, axis=2)
        return np.concatenate([dir_image, visit_image, wall_image], axis=2).astype(dtype)

    def get_visit_map_image(self):
        x, y = int(self.pos_est.px), int(self.pos_est.py)
        return self.visit_map.fetch_around(x, y)

    def get_wall_map_image(self):
        x, y = int(self.pos_est.px), int(self.pos_est.py)
        return self.wall_map.fetch_around(x, y)

    def get_direction_image(self):
        return self.dir_map.get_map(self.pos_est.direction)

    def concat_images(self):
        visit_image = self.get_visit_map_image()
        wall_image = self.get_wall_map_image()
        dir_image = self.get_direction_image()

        visit_image = np.expand_dims(visit_image, axis=2)
        wall_image = np.expand_dims(wall_image, axis=2)
        dir_image = np.expand_dims(dir_image, axis=2)
        dummy = np.zeros_like(wall_image)

        visit_image = np.concatenate([dummy, visit_image, dummy], axis=2).astype(np.float32)
        wall_image = np.concatenate([dummy, dummy, wall_image], axis=2).astype(np.float32)
        dir_image = np.concatenate([dir_image, dummy, dummy], axis=2).astype(np.float32)
        return np.concatenate([visit_image, wall_image, dir_image], axis=0)

