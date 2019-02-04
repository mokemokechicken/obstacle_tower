from collections import Counter

from obstacle_tower_env import ObstacleTowerEnv

from tower.const import Action
from tower.spike.util import average_image, frame_abs_diff
import numpy as np


class JumpCycleCheck:
    def __init__(self, env: ObstacleTowerEnv):
        self.env = env
        self.done = False
        self._last_frame = None
        self._frame_diffs = []
        self._check_cycle = 40
        self._jump_cycles = []

    def reset(self):
        self._frame_diffs = []
        self._last_frame = None

    @property
    def estimated_jump_cycle(self):
        print(self._jump_cycles)
        return Counter(self._jump_cycles).most_common(1)[0][0]

    def have_confidence(self):
        if not self._jump_cycles:
            return False
        if Counter(self._jump_cycles).most_common(1)[0][1] >= 3:  # ３つ以上同じcycleになれば
            return True

    def step(self):
        if len(self._frame_diffs) == 0:
            self.env.step(Action.JUMP)
        else:
            self.env.step(Action.NOP)
        small_frame = average_image(self.env.render())
        if self._last_frame is not None:
            diff = round(frame_abs_diff(small_frame, self._last_frame), 1)
            self._frame_diffs.append(diff)

        if len(self._frame_diffs) >= self._check_cycle:
            self._jump_cycles.append(self._estimate_cycle())
            if self.have_confidence():
                self.done = True
            else:
                self.reset()

        self._last_frame = small_frame

    def _estimate_cycle(self):
        scores = []
        start_idx = 2
        for i in range(start_idx, len(self._frame_diffs) - 1):
            # 2つに分ける。
            g1, g2 = self._frame_diffs[:i], self._frame_diffs[i:]
            # scores.append(round(np.abs(np.mean(g1) - np.mean(g2)), 1))
            scores.append(max(round(np.std(g1), 1), round(np.std(g2), 1)))

        print(scores)
        return int(np.argmin(scores)) + start_idx + 1
