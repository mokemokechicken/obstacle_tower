import numpy as np


class StateHistory:
    def __init__(self, state_size, history_size=1000):
        self.memory_size = history_size
        self.state_size = state_size
        self.index = 0
        self._memory = np.zeros((history_size, state_size), dtype=np.float32)

    @property
    def memory(self):
        return self._memory

    def store(self, state):
        index = self.index % self.memory_size
        self.memory[index, ] = state
        self.index += 1

    def difference_array(self, state):
        max_index = min(self.memory_size, self.index)
        if max_index > 0:
            return np.sum(np.square(self.memory[:max_index, ] - state), axis=1)
        else:
            return None