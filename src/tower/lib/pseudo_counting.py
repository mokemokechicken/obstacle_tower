from collections import Counter
from typing import List

import numpy as np


# https://www.slideshare.net/sotetsukoyamada/montezumas-revenge-nips2016
class PseudoCounting:
    def __init__(self, n_factors):
        self.n_factors = n_factors
        self.counters = [Counter() for _ in range(self.n_factors)]
        self.reset()

    def reset(self):
        for c in self.counters:
            c.clear()

    def add_count(self, values):
        assert len(self.counters) == len(values)
        for counter, v in zip(self.counters, values):
            counter[v] += 1

    def estimate_count(self, values):
        """
        # p1 = (1/10) * (1/10) * (9/10)
        # p2 = (2/11) * (2/11) * (10/11)
        # N = p1*(1-p2) / (p2 - p1)

        # log(p1) = log(1/10) + log(1/10) + log(9/10)
        # log(p2) = log(2/11) + log(2/11) + log(10/11)
        """
        assert len(self.counters) == len(values)

        log_p1 = 0
        log_p2 = 0
        for counter, v in zip(self.counters, values):
            total = sum(counter.values())
            target = counter[v]
            if target == 0:
                return 0
            log_p1 += np.log(target/total)
            log_p2 += np.log((target+1)/(total+1))
        p1 = np.exp(log_p1)
        p2 = np.exp(log_p2)
        if p1 == p2:
            return 1e+30
        return p1*(1-p2)/(p2-p1)


if __name__ == '__main__':
    pc = PseudoCounting(3)
    pc.add_count(["SUN", "LATE", "QUIET"])
    for _ in range(9):
        pc.add_count(["RAIN", "EARLY", "BUSY"])
    print(pc.estimate_count(["SUN", "LATE", "BUSY"]))
