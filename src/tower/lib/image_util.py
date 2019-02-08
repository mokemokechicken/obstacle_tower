import cv2
from cv2.cv2 import COLOR_BGR2HSV, COLOR_HSV2BGR
import numpy as np


def bgr_to_hsv(img, to_float=True, from_float=False):
    """

    :param img: 0~255 BGR ndarray(h, w, BGR)
    :param to_float:
    :param from_float:
    :return: ndarray(h, w, hsv), H = 0~179 . S,V = 0~255
    """
    if from_float:
        img = (img * 255).astype(dtype=np.uint8)
    ret = cv2.cvtColor(img, COLOR_BGR2HSV)
    if to_float:
        ret[:, :, 0] = ret[:, :, 0] / 180.
        ret[:, :, 1:3] = ret[:, :, 1:3] / 255.
    return ret


def hsv_to_bgr(img, from_float=True, to_float=False):
    """

    :param to_float:
    :param from_float:
    :param img: H:0~179, SV: 0~255,  HSV ndarray(h, w, HSV)
    :return: 0~255 BGR ndarray(h, w, BGR)
    """
    if from_float:
        img = np.copy(img)
        img[:, :, 0] = img[:, :, 0] * 180
        img[:, :, 1:3] = img[:, :, 1:3] * 255
        img = img.astype(dtype=np.uint8)
    ret = cv2.cvtColor(img, COLOR_HSV2BGR)
    if to_float:
        ret = ret / 255.
    return ret
