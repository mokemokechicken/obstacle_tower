import numpy as np


class Action:
    NOP = np.array([0, 0, 0, 0])
    FORWARD = np.array([1, 0, 0, 0])
    BACK = np.array([2, 0, 0, 0])
    CAMERA_LEFT = np.array([0, 1, 0, 0])
    CAMERA_RIGHT = np.array([0, 2, 0, 0])
    JUMP = np.array([0, 0, 1, 0])
    RIGHT = np.array([0, 0, 0, 1])
    LEFT = np.array([0, 0, 0, 2])