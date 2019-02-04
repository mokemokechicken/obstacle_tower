from collections import Counter
from logging import getLogger

from obstacle_tower_env import ObstacleTowerEnv

from tower.spike.util import average_image
import numpy as np

logger = getLogger(__name__)


class ActionCycleCheck:
    def __init__(self, env: ObstacleTowerEnv, action):
        self.env = env
        self.action = action
        self.done = False
        self._frames = []
        self._check_cycle = 20
        self._estimated_cycles = []

    def step(self):
        self.env.step(self.action)
        small_frame = average_image(self.env.render())
        self._frames.insert(0, small_frame)
        if len(self._frames) > self._check_cycle:
            self.estimate_cycle()
            if self.have_confidence():
                self.done = True

    def have_confidence(self):
        if not self._estimated_cycles:
            return False
        return Counter(self._estimated_cycles).most_common(1)[0][1] >= 3

    @property
    def estimated_cycle(self):
        return Counter(self._estimated_cycles).most_common(1)[0][0]

    def estimate_cycle(self):
        data = np.array(self._frames)
        diffs = []
        cycles = []
        for cycle in range(3, len(self._frames)//2):
            base_idx = 0
            dfs = []
            while base_idx+cycle*2 <= len(self._frames):
                diff = np.sum(np.abs(data[base_idx:base_idx+cycle] - data[base_idx+cycle:base_idx+cycle*2])) / cycle
                dfs.append(diff)
                base_idx += cycle

            diffs.append(float(np.mean(dfs)))
            cycles.append(cycle)

        m, s = np.mean(diffs), np.std(diffs)
        logger.info(diffs)
        diffs = [(x-m)/s for x in diffs]
        if np.min(diffs) < -1.5:
            cy = cycles[int(np.argmin(diffs))]
            logger.info(cy)
            self._estimated_cycles.append(cy)
        self._check_cycle += 10
