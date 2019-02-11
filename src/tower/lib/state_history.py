import numpy as np

from tower.lib.pseudo_counting import PseudoCounting


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
        self.pseudo_counting = PseudoCounting(state_size)

    @property
    def memory(self):
        return self._memory

    def reset(self, also_pseudo_count=False):
        self._memory[:, :] = 0.
        self.index = 0
        self.recent_rarity = 0
        self.last_rarity = 0
        self.last_state = None
        if also_pseudo_count:
            self.pseudo_counting.reset()

    def store(self, latent_state, obs):
        index = self.index % self.memory_size

        state = self._make_state(latent_state, obs)
        self._add_discrete_counting(state)
        differences = self.difference_array(state)
        rarity = float(0. if differences is None else np.min(differences))
        self.memory[index, ] = state
        self.index += 1
        self.recent_rarity = (1-self.alpha) * self.recent_rarity + self.alpha * rarity

        self.last_rarity = rarity
        self.last_state = state
        return

    @staticmethod
    def _make_state(latent_state, obs):
        return np.array(list(latent_state) + [obs[1]])

    @staticmethod
    def _convert_state_to_discrete(state):
        return tuple((state * 10).astype(np.int))

    def _add_discrete_counting(self, state):
        self.pseudo_counting.add_count(self._convert_state_to_discrete(state))

    def estimate_count(self, latent_state, obs) -> float:
        state = self._make_state(latent_state, obs)
        discrete_state = self._convert_state_to_discrete(state)
        return self.pseudo_counting.estimate_count(discrete_state)

    def difference_array(self, state):
        max_index = min(self.memory_size, self.index)
        if max_index > 0:
            return np.sum(np.square(self.memory[:max_index, ] - state), axis=1)
        else:
            return None
