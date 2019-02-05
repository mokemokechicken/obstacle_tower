import math
from logging import getLogger

from tower.const import Action
from tower.event_handlers.base import EventHandler, EventParamsAfterStep
from tower.event_handlers.moving_checker import MovingChecker
from tower.event_handlers.position_estimator import PositionEstimator

import numpy as np

logger = getLogger(__name__)


class MapObservation(EventHandler):
    def __init__(self, position_estimator: PositionEstimator, moving_checker: MovingChecker):
        self.pos_est = position_estimator
        self.moving_checker = moving_checker

        self.VISIT_SCALE = 10.
        self.VISIT_VALUE = 0.1
        self.WALL_SCALE = 1.
        self.WALL_VALUE = 0.3

        self.visit_map = MapController(size=64, name="visit", scale=self.VISIT_SCALE)
        self.wall_map = MapController(size=64, name="wall", scale=self.WALL_SCALE)

    def after_step(self, params: EventParamsAfterStep):
        self.visit_map.add_value(self.pos_est.px, self.pos_est.py, self.VISIT_VALUE)

        action = params.action
        if not self.moving_checker.did_move and (action[Action.IDX_MOVE_FB] > 0 or action[Action.IDX_MOVE_RL] > 0):
            self.wall_map.add_value(self.pos_est.px + self.pos_est.dx, self.pos_est.py + self.pos_est.dy,
                                    self.WALL_VALUE)

    def image(self):
        visit_image = self.get_visit_map_image()
        wall_image = self.get_wall_map_image()

        visit_image = np.expand_dims(visit_image, axis=2)
        wall_image = np.expand_dims(wall_image, axis=2)
        dummy = np.zeros_like(wall_image)
        return np.concatenate([wall_image, visit_image, dummy], axis=2).astype(np.float32)

    def get_visit_map_image(self):
        x, y = int(self.pos_est.px), int(self.pos_est.py)
        return self.visit_map.fetch_around(x, y)

    def get_wall_map_image(self):
        x, y = int(self.pos_est.px), int(self.pos_est.py)
        return self.wall_map.fetch_around(x, y)

    def concat_images(self):
        visit_image = self.get_visit_map_image()
        wall_image = self.get_wall_map_image()

        visit_image = np.expand_dims(visit_image, axis=2)
        wall_image = np.expand_dims(wall_image, axis=2)
        dummy = np.zeros_like(wall_image)

        visit_image = np.concatenate([dummy, visit_image, dummy], axis=2).astype(np.float32)
        wall_image = np.concatenate([dummy, dummy, wall_image], axis=2).astype(np.float32)
        return np.concatenate([visit_image, wall_image], axis=0)


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
        self.map[my, mx] = float(np.clip(self.map[my, mx] + value, self.min_value, self.max_value))
        logger.info(f"map {self.name or ''}({x},{y})={self.map[my, mx]}")

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
