import math

from tower.event_handlers.base import EventHandler, EventParamsAfterStep
from tower.const import ROTATION_CYCLE, Action
from tower.event_handlers.judge_move import JudgeMove


class PositionEstimator(EventHandler):
    def __init__(self, judger: JudgeMove):
        self.px = 0
        self.py = 0
        self.pz = 0
        self.direction = 15  # 0~19: 0 is direction=(1, 0, 0), 5 is direction=(0, 1, 0), 15 is (0, -1, 0)
        self.d_size = 1.
        self.judger = judger

    def after_step(self, params: EventParamsAfterStep):
        action = params.action
        if action[Action.IDX_CAMERA_LR] > 0:
            rl = -1 if action[Action.IDX_CAMERA_LR] == 1 else 1
            self.direction = (self.direction + rl) % ROTATION_CYCLE

        rad = (self.direction * 18) * math.pi / 180
        dx = dy = 0
        if action[Action.IDX_MOVE_FB] > 0:
            fb = 1 if action[Action.IDX_MOVE_FB] == 1 else -1
            dx += math.cos(rad) * fb
            dy += math.sin(rad) * fb

        if action[Action.IDX_MOVE_RL] > 0:
            rl = 1 if action[Action.IDX_MOVE_RL] == 1 else -1  # rl = 1 is right, -1 is left
            dx += math.cos(rad - math.pi / 2) * rl
            dy += math.sin(rad - math.pi / 2) * rl

        if self.judger.did_move:
            self.px += dx * self.d_size
            self.py += dy * self.d_size
