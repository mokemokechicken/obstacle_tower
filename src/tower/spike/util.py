import numpy as np


def average_image(frame, kernel_size=8):
    kernel = np.ones((kernel_size ** 2)).reshape((kernel_size, kernel_size)) / kernel_size ** 2
    fs = [stride_conv2d(frame[:, :, i], kernel, kernel_size) for i in range(frame.shape[2])]
    return np.transpose(np.array(fs), (1, 2, 0))


def frame_pixel_diff(f1, f2):
    return np.mean(np.abs(f1 - f2), axis=2)


def frame_abs_diff(f1, f2) -> float:
    return float(np.sum(np.abs(f1 - f2)))


def stride_conv2d(img: np.ndarray, kernel: np.ndarray, stride: int = 1):
    """

    :param img:  2d-array
    :param kernel: 2d-array
    :param int stride: stride
    :return:
    """
    s1, s2 = img.strides
    i1, i2 = img.shape
    k1, k2 = kernel.shape
    out_shape = (k1, k2, 1 + (i1 - k1) // stride, 1 + (i2 - k2) // stride)
    stride_array = np.lib.stride_tricks.as_strided(img, shape=out_shape, strides=(s1, s2, s1 * stride, s2 * stride))
    return np.einsum('ij,ijkl->kl', kernel, stride_array)


def to_onehot(idx, size):
    ret = [0] * size
    ret[idx] = 1
    return ret
