from collections import Counter
from logging import basicConfig, INFO
from pathlib import Path
from typing import List

import cv2
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from obstacle_tower_env import ObstacleTowerEnv
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.optimizers import Adam

from tower.spike.action_cycle_check import ActionCycleCheck
from tower.spike.const import Action
from tower.spike.jump_cycle_check import JumpCycleCheck
from tower.spike.moving_check_agent import MovingCheckAgent, CheckResult

PRJ_ROOT = Path(__file__).parents[3]


def main():
    basicConfig(level=INFO)
    env = ObstacleTowerEnv(str(PRJ_ROOT / 'obstacletower'), retro=False, worker_id=9)
    done = False
    frames = []
    env.floor(1)
    env.reset()

    screen = Screen()
    mc_agent = MovingCheckAgent(env)
    jc_agent = ActionCycleCheck(env, Action.JUMP)

    for _ in range(20):
        env.step(Action.FORWARD)

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

        # obs, reward, done, info = env.step(cc.action)
        # cc.prev_small_image(small_image)

        jc_agent.step()
        if jc_agent.done:
            print(jc_agent.estimated_cycle)
            break

        # mc_agent.step()
        # if mc_agent.done:
        #     instant_training(mc_agent.results)
        #     break


def instant_training(results: List[CheckResult]):
    frame_shape = list(results[0].frame0.shape)
    frame_shape[-1] *= 2
    input_f01 = Input(shape=frame_shape)
    x = Conv2D(32, kernel_size=8, strides=4, padding="valid", activation='relu')(input_f01)
    x = Flatten()(x)
    forward_back = Dense(3, activation='softmax', name="move_forward_back")(x)
    move_left_right = Dense(3, activation='softmax', name="move_left_right")(x)
    camera_left_right = Dense(3, activation='softmax', name="camera_left_right")(x)
    up_down = Dense(3, activation='softmax', name="up_down")(x)
    model = Model(input_f01, [forward_back, camera_left_right, up_down, move_left_right])
    model.compile(Adam(lr=0.001), loss=categorical_crossentropy)

    data_x = []
    data_y = []
    for result in results:
        data_x.append(np.concatenate([result.frame0, result.frame1], axis=2))
        action = result.action
        mfb_true = to_onehot(action[0], 3)
        rlr_true = to_onehot(action[1], 3)
        ud_true = to_onehot(action[2], 3)
        mlr_true = to_onehot(action[3], 3)
        data_y.append([mfb_true, rlr_true, ud_true, mlr_true])

    train_data_x = np.vstack([np.expand_dims(x, axis=0) for x in data_x])
    train_data_y = [np.array([x[i] for x in data_y]) for i in range(4)]

    while True:
        model.fit(train_data_x, train_data_y, epochs=10)
        preds = model.predict(train_data_x)
        counter = Counter()
        for pred, true_y in zip(preds, train_data_y):  # forward/back, camera, up_down, left/right
            for di in range(len(pred)):
                idx = np.argmax(pred[di])
                counter["total"] += 1
                if true_y[di, idx] == 1:
                    counter["ok"] += 1
        print(f"accuracy: {100 * counter['ok'] / counter['total']:.1f}% ({counter['ok']}/{counter['total']})")


def to_onehot(idx, size):
    ret = [0] * size
    ret[idx] = 1
    return ret


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


def output_to_movie(frames):
    fig = plt.gcf()
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
    anim.save("tower.mp4", writer='ffmpeg')
