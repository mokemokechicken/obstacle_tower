import numpy as np


class StateHistory:
    def __init__(self, state_size, history_size=1000, alpha=0.05):
        self.memory_size = history_size
        self.state_size = state_size
        self.index = 0
        self._memory = np.zeros((history_size, state_size), dtype=np.float32)
        self.last_rarity = 0
        self.recent_rarity = 0
        self.alpha = alpha
        self.last_state = None

    @property
    def memory(self):
        return self._memory

    def reset(self):
        self._memory[:, :] = 0.
        self.index = 0
        self.recent_rarity = 0
        self.last_rarity = 0
        self.last_state = None

    def store(self, latent_state, obs):
        index = self.index % self.memory_size

        state = np.array(list(latent_state) + [obs[1]])
        differences = self.difference_array(state)
        rarity = float(0. if differences is None else np.min(differences))
        self.memory[index, ] = state
        self.index += 1
        self.recent_rarity = (1-self.alpha) * self.recent_rarity + self.alpha * rarity

        self.last_rarity = rarity
        self.last_state = state
        return

    def difference_array(self, state):
        max_index = min(self.memory_size, self.index)
        if max_index > 0:
            return np.sum(np.square(self.memory[:max_index, ] - state), axis=1)
        else:
            return None
