from logging import getLogger

import numpy as np
from scipy import stats

from tower.const import Action
from tower.observation.base import EventHandler, EventParamsAfterStep
from tower.observation.frame import FrameHistory
from tower.spike.util import frame_abs_diff

logger = getLogger(__name__)


class MovingChecker(EventHandler):
    def __init__(self, frame_history: FrameHistory):
        self.frame_history = frame_history
        self._nop_diffs = []
        self._other_diffs = []
        self._step = 0
        self._nop_dist = None
        self._other_dist = None
        self._last_jump_counter = 0
        self._did_move = True

    @property
    def did_move(self):
        return self._did_move

    def after_step(self, params: EventParamsAfterStep):
        action = params.action
        prev_frame = self.frame_history.last_small_frame
        cur_frame = self.frame_history.current_small_frame
        #
        is_nop_action = tuple(action) == tuple(Action.NOP)
        diff = frame_abs_diff(prev_frame, cur_frame)
        add_history = True

        self._step += 1
        if self._step == 10:
            self.update_dist()
            self._step = 0

        # if jumped, frame_diff will not be added to history after N(=20) frames.
        if Action.is_jump_action(action):
            self._last_jump_counter = 20
        if self._last_jump_counter > 0:
            self._last_jump_counter -= 1

        if not is_nop_action:
            if self._nop_dist is not None:
                p_nop = self._nop_dist.pdf(diff)
                p_other = self._other_dist.pdf(diff)
                # logger.info(f"P_nop={p_nop}, P_other={p_other}")
                is_nop_action = p_other < p_nop
                # if is_nop_action:
            else:
                is_nop_action = diff < 3
            if is_nop_action:
                add_history = False
                logger.info(f"action={action}, but did not move: diff={diff}")

        if is_nop_action:
            if add_history and self._last_jump_counter == 0:
                # logger.info(f"add diff to NOP Diff List: {diff}")
                self._nop_diffs.append(diff)
                self._nop_diffs = self._nop_diffs[-100:]
        else:
            # logger.info(f"add diff to OTHER Diff List: {diff}")
            self._other_diffs.append(diff)
            self._other_diffs = self._other_diffs[-100:]
        self._did_move = not is_nop_action

    def update_dist(self):
        if len(self._nop_diffs) > 3 and len(self._other_diffs) > 3:
            self._nop_dist = stats.norm(np.mean(self._nop_diffs), np.std(self._nop_diffs) + 0.1)
            self._other_dist = stats.norm(np.mean(self._other_diffs), np.std(self._other_diffs))
