import numpy as np


ROTATION_CYCLE = 20


class Action:
    size = 54
    NOP = np.array([0, 0, 0, 0])
    FORWARD = np.array([1, 0, 0, 0])
    BACK = np.array([2, 0, 0, 0])
    CAMERA_LEFT = np.array([0, 1, 0, 0])
    CAMERA_RIGHT = np.array([0, 2, 0, 0])
    JUMP = np.array([0, 0, 1, 0])
    RIGHT = np.array([0, 0, 0, 1])
    LEFT = np.array([0, 0, 0, 2])

    IDX_MOVE_FB = 0
    IDX_CAMERA_LR = 1
    IDX_JUMP = 2
    IDX_MOVE_RL = 3

    @staticmethod
    def sample_action():
        return Action.from_int(int(np.random.randint(0, Action.size)))

    @staticmethod
    def to_int(action) -> int:
        return int(action[0]*(3*2*3) + action[1]*(2*3) + action[2]*3 + action[3])

    @staticmethod
    def from_int(idx: int):
        """

        :param idx:
        :return:

        >>> all([Action.to_int(Action.from_int(x)) == x for x in range(54)])
        True
        """
        assert 0 <= idx < 54
        action = np.zeros((4, ), dtype=np.int8)
        action[0] = idx // (3*2*3)
        action[1] = (idx % (3*2*3)) // (2*3)
        action[2] = (idx % (2*3)) // 3
        action[3] = idx % 3
        return action

    @staticmethod
    def is_jump_action(action):
        return action[Action.IDX_JUMP] == 1
