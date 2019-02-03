from logging import getLogger

from scipy import stats

from tower.spike.const import Action
from tower.spike.util import frame_abs_diff
import numpy as np

logger = getLogger(__name__)


class JudgeMove:
    def __init__(self):
        self._nop_diffs = []
        self._other_diffs = []
        self._step = 0
        self._nop_dist = None
        self._other_dist = None

    def did_move(self, prev_frame, cur_frame, action) -> bool:
        is_nop_action = tuple(action) == tuple(Action.NOP)
        diff = frame_abs_diff(prev_frame, cur_frame)
        add_history = True

        self._step += 1
        if self._step == 10:
            self.update_dist()
            self._step = 0

        if not is_nop_action:
            if self._nop_dist is not None:
                p_nop = self._nop_dist.pdf(diff)
                p_other = self._other_dist.pdf(diff)
                logger.info(f"P_nop={p_nop}, P_other={p_other}")
                is_nop_action = p_other < p_nop
                # if is_nop_action:
            else:
                is_nop_action = diff < 3
            if is_nop_action:
                add_history = False
                logger.info(f"action={action}, but did not move")

        if is_nop_action:
            if add_history:
                logger.info(f"add diff to NOP: {diff}")
                self._nop_diffs.append(diff)
                self._nop_diffs = self._nop_diffs[-100:]
        else:
            logger.info(f"add diff to OTHER: {diff}")
            self._other_diffs.append(diff)
            self._other_diffs = self._other_diffs[-100:]
        return not is_nop_action

    def update_dist(self):
        if len(self._nop_diffs) > 3 and len(self._other_diffs) > 3:
            self._nop_dist = stats.norm(np.mean(self._nop_diffs), np.std(self._nop_diffs))
            self._other_dist = stats.norm(np.mean(self._other_diffs), np.std(self._other_diffs))
