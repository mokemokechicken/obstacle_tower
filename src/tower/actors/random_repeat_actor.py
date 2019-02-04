import numpy as np

from tower.const import Action


class RandomRepeatActor:
    action = None
    continue_rate = None
    schedules = None
    schedule_count = 0

    def __init__(self, action=None, continue_rate=0.95):
        self.reset(action, continue_rate)

    def reset(self, action=None, continue_rate=None, schedules=None):
        if continue_rate is not None:
            self.continue_rate = continue_rate
        self.action = action
        if schedules:
            self.schedules = schedules
            self.schedule_count = 0
            self.scheduled_action()

    def scheduled_action(self):
        if self.schedules:
            self.action = self.schedules[0][0]
            self.schedule_count += 1
            if self.schedule_count >= self.schedules[0][1]:
                self.schedules = self.schedules[1:]
                self.schedule_count = 0

    def decide_action(self):
        if self.schedules:
            self.scheduled_action()
        elif self.action is None or np.random.random() >= self.continue_rate:
            self.action = self.sample_action()
        return self.action

    def sample_action(self):
        r1 = np.random.random()

        action = Action.NOP

        if r1 < 0.03:
            return action

        if np.random.random() < 0.2:
            action = Action.JUMP

        if r1 < 0.6:
            action = action + Action.FORWARD
        elif r1 < 0.7:
            action = action + Action.LEFT
        elif r1 < 0.8:
            action = action + Action.RIGHT
        elif r1 < 0.85:
            action = action + Action.CAMERA_LEFT
        elif r1 < 0.95:
            action = action + Action.CAMERA_RIGHT
        else:
            action = action + Action.BACK
        return action

    @staticmethod
    def restrict_action(action):
        if action[Action.IDX_MOVE_FB] + action[Action.IDX_MOVE_RL] > 0 and action[Action.IDX_CAMERA_LR] > 0:
            if np.random.random() < 0.5:
                action[Action.IDX_CAMERA_LR] = 0
            else:
                action[Action.IDX_MOVE_FB] = 0
                action[Action.IDX_MOVE_RL] = 0
