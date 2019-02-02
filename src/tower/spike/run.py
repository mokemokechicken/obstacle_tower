from pathlib import Path

import cv2
from PIL import Image
from PIL.Image import BOX
from matplotlib import animation
from matplotlib import pyplot as plt
from obstacle_tower_env import ObstacleTowerEnv

import numpy as np

PRJ_ROOT = Path(__file__).parents[3]


def main():
    env = ObstacleTowerEnv(str(PRJ_ROOT / 'obstacletower'), retro=False, worker_id=9)
    done = False
    frames = []
    env.reset()
    cv2.namedWindow("original")
    cv2.moveWindow("original", 0, 0)
    cv2.resizeWindow("original", 240, 200)

    cv2.namedWindow("mean")
    cv2.moveWindow("mean", 400, 0)
    cv2.resizeWindow("mean", 50, 50)

    cv2.namedWindow("diff")
    cv2.moveWindow("diff", 800, 0)
    cv2.resizeWindow("diff", 50, 50)

    last_av_image = None

    while not done:
        frame = env.render()
        frames.append(frame)
        av_image = average_image(frame)
        cv2.imshow("original", frame)
        cv2.imshow("mean", av_image)
        if last_av_image is not None:
            diff_av_image = np.mean(np.abs(av_image - last_av_image), axis=2)
            cv2.imshow("diff", diff_av_image)
        cv2.waitKey(0)
        last_av_image = av_image
        # obs, reward, done, info = env.step(env.action_space.sample())
        obs, reward, done, info = env.step([1, 0, 0, 0])
        print(f"reward={reward}")


def average_image(frame, kernel_size=8):
    kernel = np.ones((kernel_size**2)).reshape((kernel_size, kernel_size)) / kernel_size**2
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
    main()
