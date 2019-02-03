from pathlib import Path

import cv2
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from obstacle_tower_env import ObstacleTowerEnv

from tower.spike.const import Action
from tower.spike.moving_check_agent import MovingCheckAgent

PRJ_ROOT = Path(__file__).parents[3]


def main():
    env = ObstacleTowerEnv(str(PRJ_ROOT / 'obstacletower'), retro=False, worker_id=9)
    done = False
    frames = []
    env.floor(1)
    env.reset()

    screen = Screen()
    mc_agent = MovingCheckAgent(env)

    while not done:
        frame = env.render()
        frames.append(frame)
        screen.show("original", frame)

        # small_image = average_image(frame)
        # screen.show("mean", small_image)
        # if last_av_image is not None:
        #     diff_av_image = np.mean(np.abs(avg_image - last_av_image), axis=2)
        #     screen.show("diff", diff_av_image)
        # last_av_image = avg_image
        cv2.waitKey(0)
        # obs, reward, done, info = env.step(env.action_space.sample())

        # obs, reward, done, info = env.step(rotation_test.act)
        mc_agent.step()
        if mc_agent.done:
            break


class RotationTest:
    def __init__(self):
        self.smalls = []
        self.act = Action.CAMERA_RIGHT
        self.min_cycle = 10
        self.max_cycle = 100

    def prev_small_image(self, small_image):
        self.smalls.insert(0, small_image.flatten())
        if len(self.smalls) > self.max_cycle:
            self.smalls = self.smalls[:self.max_cycle]

        data = np.vstack(self.smalls)
        diffs = {}
        for cycle in range(self.min_cycle, len(self.smalls) // 2):
            diff = round(np.sum(np.abs(data[:cycle, ] - data[cycle:cycle * 2, ])) * 10)
            print(f"cycle={cycle}: {diff}")
            diffs[cycle] = diff

        if len(diffs) > 2:
            vs = list(diffs.values())
            mean, std = np.mean(vs), np.std(vs)
            for k, v in diffs.items():
                sv = (v - mean) / std
                if sv < -1:
                    print(f"cycle={k}, {sv}")


ROTATION_CYCLE = 20


class Screen:
    def __init__(self):
        self._windows = {}

    def show(self, window_name, image):
        if window_name not in self._windows:
            self.setup_window(window_name)
        cv2.imshow(window_name, image)

    def setup_window(self, window_name):
        idx = len(self._windows)
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, (idx % 3) * 400, (idx // 3) * 300)
        self._windows[window_name] = 1


def average_image(frame, kernel_size=8):
    kernel = np.ones((kernel_size ** 2)).reshape((kernel_size, kernel_size)) / kernel_size ** 2
    fs = [stride_conv2d(frame[:, :, i], kernel, kernel_size) for i in range(frame.shape[2])]
    return np.transpose(np.array(fs), (1, 2, 0))


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


def output_to_movie(frames):
    fig = plt.gcf()
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
    anim.save("tower.mp4", writer='ffmpeg')


if __name__ == '__main__':
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    main()

"""
# MOVE: [NO, forword, back]
# CAMERA: [NO, rotate left, rotate right]
# Jump: [NO, jump]
# MOVE: [NO, right, left]
"""
