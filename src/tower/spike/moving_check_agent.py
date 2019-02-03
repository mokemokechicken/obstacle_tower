from collections import namedtuple, Counter
import numpy as np
from obstacle_tower_env import ObstacleTowerEnv
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.optimizers import Adam

from tower.spike.const import Action
from tower.spike.util import average_image, frame_abs_diff, to_onehot

CheckResult = namedtuple('CheckResult', 'frame0 frame1 action')


class MovingCheckAgent:
    def __init__(self, env: ObstacleTowerEnv):
        self.env = env
        self.n_step = 20
        self.check_actions = [Action.NOP, Action.FORWARD, Action.BACK,
                              Action.CAMERA_RIGHT, Action.LEFT,
                              Action.RIGHT, Action.LEFT]
        self.state = CheckState()
        self.results = []
        self._last_small_frame = None
        self._model = None

    @property
    def done(self):
        return self.state.done

    def reset(self):
        self.state = CheckState()
        self.results = []

    def step(self):
        if self.done:
            return

        action = self.check_actions[self.state.action_index]
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
            action = Action.NOP
        self._last_small_frame = small_frame
        self.results.append(CheckResult(frame0, frame1, action))

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
        x = Flatten()(x)
        forward_back = Dense(3, activation='softmax', name="move_forward_back")(x)
        move_left_right = Dense(3, activation='softmax', name="move_left_right")(x)
        camera_left_right = Dense(3, activation='softmax', name="camera_left_right")(x)
        up_down = Dense(3, activation='softmax', name="up_down")(x)
        model = Model(input_f01, [forward_back, camera_left_right, up_down, move_left_right])
        model.compile(Adam(lr=0.001), loss=categorical_crossentropy)
        return model

    def update_model(self):
        data_x = []
        data_y = []
        for result in self.results:
            data_x.append(np.concatenate([result.frame0, result.frame1], axis=2))
            action = result.action
            mfb_true = to_onehot(action[0], 3)
            rlr_true = to_onehot(action[1], 3)
            ud_true = to_onehot(action[2], 3)
            mlr_true = to_onehot(action[3], 3)
            data_y.append([mfb_true, rlr_true, ud_true, mlr_true])

        train_data_x = np.vstack([np.expand_dims(x, axis=0) for x in data_x])
        train_data_y = [np.array([x[i] for x in data_y]) for i in range(4)]

        accuracy = self.check_accuracy(train_data_x, train_data_y)
        while accuracy < 0.9:
            self.model.fit(train_data_x, train_data_y, epochs=5)
            accuracy = self.check_accuracy(train_data_x, train_data_y)

    def check_accuracy(self, x, y):
        preds = self.model.predict(x)
        counter = Counter()
        for pred, true_y in zip(preds, y):  # forward/back, camera, up_down, left/right
            for di in range(len(pred)):
                idx = np.argmax(pred[di])
                counter["total"] += 1
                if true_y[di, idx] == 1:
                    counter["ok"] += 1
        accuracy = counter['ok'] / counter['total']
        print(f"accuracy: {100 * accuracy:.1f}% ({counter['ok']}/{counter['total']})")
        return accuracy



class CheckState:
    action_index: int = 0
    current_step: int = 0
    done: bool = False
