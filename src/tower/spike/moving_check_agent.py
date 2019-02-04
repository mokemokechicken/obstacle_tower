import random
from collections import namedtuple, Counter
from logging import getLogger

import numpy as np
from obstacle_tower_env import ObstacleTowerEnv
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.optimizers import Adam

from tower.const import Action
from tower.spike.util import average_image, frame_abs_diff, to_onehot

CheckResult = namedtuple('CheckResult', 'frame0 frame1 action')

logger = getLogger(__name__)


class MovingCheckAgent:
    def __init__(self, env: ObstacleTowerEnv):
        self.env = env
        self.n_step = 5
        self.check_actions = [Action.NOP, Action.FORWARD, Action.BACK,
                              Action.CAMERA_RIGHT, Action.CAMERA_LEFT,
                              Action.RIGHT, Action.LEFT,
                              Action.FORWARD, Action.CAMERA_RIGHT]
        self.state = CheckState()
        self.results = []
        self._last_small_frame = None
        self._model = None

    @property
    def done(self):
        return self.state.done

    def reset(self):
        self.n_step = int(random.random()*4) + 4
        self.state = CheckState()
        self.results = self.results[-1000:]

    def step(self):
        if self.done:
            return

        recording_action = action = self.check_actions[self.state.action_index]
        self.state.current_step += 1
        if self.state.current_step == self.n_step:
            self.state.current_step = 0
            self.state.action_index += 1
            if self.state.action_index >= len(self.check_actions):
                self.state.done = True

        frame0 = self.env.render()
        if self._last_small_frame is None:
            self._last_small_frame = average_image(frame0)
        self.env.step(action)
        frame1 = self.env.render()
        small_frame = average_image(frame1)
        if frame_abs_diff(self._last_small_frame, small_frame) < 3:
            logger.info("convert to NOP")
            recording_action = Action.NOP
        self._last_small_frame = small_frame
        self.results.append(CheckResult(frame0, frame1, recording_action))
        return action

    @property
    def model(self) -> Model:
        if self._model is None:
            self._model = self.build_model()
        return self._model

    def build_model(self):
        frame_shape = list(self.results[0].frame0.shape)
        frame_shape[-1] *= 2
        input_f01 = Input(shape=frame_shape)
        x = Conv2D(32, kernel_size=8, strides=4, padding="valid", activation='relu')(input_f01)
        x = Conv2D(32, kernel_size=3, strides=1, padding="valid", activation='relu')(x)
        x = Flatten()(x)
        # forward_back = Dense(4, activation='softmax', name="move_forward_back")(x)
        # move_left_right = Dense(3, activation='softmax', name="move_left_right")(x)
        # camera_left_right = Dense(3, activation='softmax', name="camera_left_right")(x)
        # up_down = Dense(3, activation='softmax', name="up_down")(x)
        # model = Model(input_f01, [forward_back, camera_left_right, up_down, move_left_right])
        x = Dense(24, activation='relu')(x)
        actions = Dense(54, activation='softmax', name="actions")(x)
        model = Model(input_f01, actions)
        model.compile(Adam(lr=0.001), loss=categorical_crossentropy)
        return model

    def confirm_action(self, action):
        px = self.prepare_x(n=1)
        pred = self.model.predict(px)[0]

        p_idx = int(np.argmax(pred))
        a_idx = Action.to_int(action)
        if p_idx == a_idx:
            logger.info(f"maybe success action={action}")

    def update_model(self):
        train_data_x = self.prepare_x()
        train_data_y = self.prepare_y()

        accuracy = self.check_accuracy(train_data_x[-30:], train_data_y[-30:])
        while accuracy < 0.9:
            self.model.fit(train_data_x, train_data_y, epochs=3)
            accuracy = self.check_accuracy(train_data_x, train_data_y)

    def prepare_x(self, n=None):
        data_x = []
        n = n or len(self.results)
        for result in self.results[:n]:
            data_x.append(np.concatenate([result.frame0, result.frame1], axis=2))
        train_data_x = np.vstack([np.expand_dims(x, axis=0) for x in data_x])
        return train_data_x

    def prepare_y(self, n=None):
        data_y = []
        n = n or len(self.results)
        for result in self.results[:n]:
            action = result.action
            data_y.append(to_onehot(Action.to_int(action), 54))
        train_data_y = np.array(data_y)
        return train_data_y

    def check_accuracy(self, x, y):
        preds = self.model.predict(x)
        counter = Counter()
        for di in range(len(preds)):
            counter['total'] += 1
            p_idx = int(np.argmax(preds[di]))
            t_idx =  int(np.argmax(y[di]))
            if p_idx == t_idx:
                counter['ok'] += 1

        accuracy = counter['ok'] / counter['total']
        logger.info(f"accuracy: {100 * accuracy:.1f}% ({counter['ok']}/{counter['total']})")
        return accuracy


class CheckState:
    action_index: int = 0
    current_step: int = 0
    done: bool = False
