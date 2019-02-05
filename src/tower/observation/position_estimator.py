import math
from logging import getLogger

from tower.const import ROTATION_CYCLE, Action
from tower.observation.base import EventHandler, EventParamsAfterStep
from tower.observation.moving_checker import MovingChecker

logger = getLogger(__name__)


class PositionEstimator(EventHandler):
    def __init__(self, judger: MovingChecker):
        self.px = 0
        self.py = 0
        self.pz = 0
        self.dx = 0
        self.dy = 0
        self.direction = 15  # 0~19: 0 is direction=(1, 0, 0), 5 is direction=(0, 1, 0), 15 is (0, -1, 0)
        self.d_size = 1.  # 1行動毎の論理的な移動幅
        self.judger = judger
        self.last_action = None

    def after_step(self, params: EventParamsAfterStep):
        self.last_action = action = params.action
        if action[Action.IDX_CAMERA_LR] > 0:
            lr = 1 if action[Action.IDX_CAMERA_LR] == 1 else -1  # camera 1 is left -1 is right
            self.direction = (self.direction + lr) % ROTATION_CYCLE

        rad = (self.direction * 18) * math.pi / 180
        dx = dy = 0

        if self.judger.did_move:
            if action[Action.IDX_MOVE_FB] > 0:
                fb = 1 if action[Action.IDX_MOVE_FB] == 1 else -1
                dx += math.cos(rad) * fb
                dy += math.sin(rad) * fb

            if action[Action.IDX_MOVE_RL] > 0:
                rl = 1 if action[Action.IDX_MOVE_RL] == 1 else -1  # rl = 1 is right, -1 is left
                dx += math.cos(rad - math.pi / 2) * rl
                dy += math.sin(rad - math.pi / 2) * rl

        if dx != 0 or dy != 0:
            d = math.sqrt(dx**2 + dy**2)
            dx /= d
            dy /= d

        self.px += dx * self.d_size
        self.py += dy * self.d_size
        self.dx = dx
        self.dy = dy

        logger.info(self.str_info())

    def str_info(self):
        ret = f"pos=({self.px:.1f},{self.py:.1f},{self.pz:.1f}), d=({self.dx:.1f},{self.dy:.1f})"
        if self.last_action is not None:
            action = self.last_action
            msg = []
            if action[Action.IDX_CAMERA_LR] > 0:
                msg.append('camera-left' if action[Action.IDX_CAMERA_LR] == 1 else 'camera-right')
            if action[Action.IDX_MOVE_FB] > 0:
                msg.append('forward' if action[Action.IDX_MOVE_FB] == 1 else 'back')

            if action[Action.IDX_MOVE_RL] > 0:
                msg.append('right' if action[Action.IDX_MOVE_RL] == 1 else 'left')  # rl = 1 is right, -1 is left

            if action[Action.IDX_JUMP] > 0:
                msg.append('jump')

            if not msg:
                msg.append('NOP')

            ret += ", " + " | ".join(msg)
        return ret
