import cv2
from cv2.cv2 import COLOR_BGR2HSV, COLOR_HSV2BGR


def bgr_to_hsv(img):
    """

    :param img: 0~255 BGR ndarray(h, w, BGR)
    :return: 0~255 HSV ndarray(h, w, hsv)
    """
    return cv2.cvtColor(img, COLOR_BGR2HSV)


def hsv_to_bgr(img):
    """

    :param img: 0~255 HSV ndarray(h, w, HSV)
    :return: 0~255 BGR ndarray(h, w, BGR)
    """
    return cv2.cvtColor(img, COLOR_HSV2BGR)
