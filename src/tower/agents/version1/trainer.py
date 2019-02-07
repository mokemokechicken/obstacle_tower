import math
from collections import namedtuple
from logging import getLogger
from typing import Set, List

from h5py.h5t import defaultdict

from tower.agents.version1.state_model import StateModel
from tower.const import Action
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
        state_model.build()
        state_model.compile()

        self.memory = FileMemory(self.config)

        all_episode_list = list(self.memory.episodes())
        np.random.shuffle(all_episode_list)
        ep_size = len(all_episode_list)
        ep_batch_size = ep_size // int(math.ceil(ep_size / self.config.train.max_episode_in_one_time))
        batch_num = ep_size // ep_batch_size

        for bi in range(batch_num):
            logger.info(f"StateModel Training {bi+1}/{batch_num}")
            episode_list = all_episode_list[bi*ep_batch_size:(bi+1)*ep_batch_size]
            training_data = self.make_training_data(episode_list)
            data_size = len(training_data.frame)
            idx_list = np.random.choice(range(data_size), p=training_data.importance, size=data_size * 10)
            state_model.fit(training_data.frame[idx_list], training_data.next_frame[idx_list],
                            training_data.action[idx_list])

    def make_training_data(self, episode_list):
        tc = self.config.train

        episodes = self.memory.load_episodes(episode_list)
        training_data = defaultdict(lambda: [])
        for episode_data in episodes:
            steps = episode_data["steps"]
            rewards = np.array([step['reward'] for step in steps])
            map_rewards = np.array([step['map_reward'] for step in steps])
            for t, (step, next_step) in enumerate(zip(steps[:-1], steps[1:])):
                training_data['frame'].append(step['state'][0] / 255.)
                training_data['next_frame'].append(next_step['state'][0] / 255.)
                training_data['action'].append(to_onehot(Action.to_int(step['action']), Action.size))
                reward = np.sum(rewards[t:t + tc.importance_step])
                map_reward = np.sum(map_rewards[t:t + tc.importance_step])
                training_data['importance'].append(reward + tc.map_reward_weight * map_reward)

        frame = np.array(training_data['frame'])
        next_frame = np.array(training_data['next_frame'])
        action = np.array(training_data['action'])
        importance = np.array(training_data['importance'])

        importance = (importance - np.mean(importance)) / np.std(importance)
        importance = np.exp(importance) / np.sum(np.exp(importance))
        return TrainingData(frame=frame, next_frame=next_frame, action=action, importance=importance)
