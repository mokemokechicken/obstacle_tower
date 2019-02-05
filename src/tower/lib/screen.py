import cv2
from matplotlib import animation
from matplotlib import pyplot as plt


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


def output_to_movie(frames):
    fig = plt.gcf()
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
    anim.save("tower.mp4", writer='ffmpeg')
