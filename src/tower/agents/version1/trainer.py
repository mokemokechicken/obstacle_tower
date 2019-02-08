import math
from collections import namedtuple, defaultdict
from logging import getLogger

import PIL
import cv2
from PIL import Image
from PIL.Image import Image
from cv2.cv2 import COLOR_BGR2HSV
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from tower.agents.version1.state_model import StateModel
from tower.const import Action
from tower.lib.image_util import bgr_to_hsv
from tower.lib.memory import FileMemory
from tower.spike.util import to_onehot
from ..base import TrainerBase
import numpy as np

# データを取り出して
# tensorboard とのやりとりは、ここがやるべき
# 打ち切りとかもね。
# state_model.update(<data>) くらいがよいIFか
# 前処理はどっちがやる？


TrainingData = namedtuple('TrainingData', 'frame next_frame action importance')

logger = getLogger(__name__)


class Trainer(TrainerBase):
    memory: FileMemory

    def train(self):
        state_model = StateModel(self.config)

        if self.config.train.new_model:
            state_model.build()
        else:
            state_model.load_model()

        state_model.compile()

        generator = self.episode_generator()
        vae = self.config.train.vae
        callbacks = [
            ReduceLROnPlateau(factor=vae.lr_decay_factor, patience=vae.lr_patience, min_lr=vae.lr_min,
                              monitor='loss', verbose=1),
        ]
        state_model.model.training_model.fit_generator(generator, steps_per_epoch=vae.steps_per_epoch,
                                                       epochs=vae.epochs, callbacks=callbacks)
        state_model.save_model()

    def episode_generator(self):
        self.memory = FileMemory(self.config)

        all_episode_list = list(self.memory.episodes())
        ep_size = len(all_episode_list)
        ep_batch_size = ep_size // int(math.ceil(ep_size / self.config.train.max_episode_in_one_time))
        batch_num = ep_size // ep_batch_size
        logger.info(f"{ep_size} episodes found")
        vc = self.config.train.vae

        training_data = None

        while True:
            np.random.shuffle(all_episode_list)
            for bi in range(batch_num):
                episode_list = all_episode_list[bi*ep_batch_size:(bi+1)*ep_batch_size]
                if training_data is None:
                    training_data = self.make_training_data(episode_list)
                data_size = len(training_data.frame)
                for ci in range(data_size // vc.batch_size):
                    idx_list = np.random.choice(range(data_size), p=training_data.importance, size=vc.batch_size)
                    frame = training_data.frame[idx_list]
                    next_frame = training_data.next_frame[idx_list]
                    action = training_data.action[idx_list]

                    targets = np.concatenate([frame, next_frame], axis=-1)
                    yield ([frame, action], targets)

                if batch_num > 1:
                    del training_data
                    training_data = None

    def make_training_data(self, episode_list):
        # logger.info(f"preparing training data")
        tc = self.config.train

        episodes = self.memory.load_episodes(episode_list)
        training_data = defaultdict(lambda: [])
        for episode_data in episodes:
            steps = episode_data["steps"]
            rewards = np.array([step['reward'] for step in steps])
            map_rewards = np.array([step['map_reward'] for step in steps])
            for t, (step, next_step) in enumerate(zip(steps[:-1], steps[1:])):
                training_data['frame'].append(bgr_to_hsv(step['state'][0]) / 255.)
                training_data['next_frame'].append(bgr_to_hsv(next_step['state'][0]) / 255.)
                training_data['action'].append(to_onehot(step['action'], Action.size))
                reward = np.sum(rewards[t:t + tc.importance_step])
                map_reward = np.sum(map_rewards[t:t + tc.importance_step])
                training_data['importance'].append(reward + tc.map_reward_weight * map_reward)

        frame = np.array(training_data['frame'])
        next_frame = np.array(training_data['next_frame'])
        action = np.array(training_data['action'])
        importance = np.array(training_data['importance'])

        importance = (importance - np.mean(importance)) / np.std(importance)
        importance = np.exp(importance) / np.sum(np.exp(importance))
        # logger.info(f"loaded {len(frame)} frames")
        return TrainingData(frame=frame, next_frame=next_frame, action=action, importance=importance)

