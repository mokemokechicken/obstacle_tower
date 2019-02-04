import numpy as np

from tower.const import Action


class RandomRepeatActor:
    action = None
    continue_rate = None

    def __init__(self, action=None, continue_rate=0.95):
        self.reset(action, continue_rate)

    def reset(self, action=None, continue_rate=None):
        if continue_rate is not None:
            self.continue_rate = continue_rate
        self.action = action

    def decide_action(self):
        if self.action is None or np.random.random() >= self.continue_rate:
            self.action = Action.sample_action()
            self.to_move_or_camera(self.action)
        return self.action

    @staticmethod
    def to_move_or_camera(action):
        if action[Action.IDX_MOVE_FB] + action[Action.IDX_MOVE_RL] > 0 and action[Action.IDX_CAMERA_LR] > 0:
            if np.random.random() < 0.5:
                action[Action.IDX_CAMERA_LR] = 0
            else:
                action[Action.IDX_MOVE_FB] = 0
                action[Action.IDX_MOVE_RL] = 0
