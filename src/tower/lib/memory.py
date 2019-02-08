import gzip
import pickle
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from tower.config import Config

logger = getLogger(__name__)


class Memory:
    def setup(self):
        pass

    def store(self, data):
        pass

    def episodes(self) -> Iterator:
        pass


class FileMemory(Memory):
    def __init__(self, config: Config):
        self.config = config
        self.base_dir = self.config.resource.memory_dir

    def setup(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def store(self, data):
        dt = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = self.base_dir / f"{dt}_{uuid4().hex}.pkl.gz"

        with gzip.open(path, mode="wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def episodes(self) -> Iterator:
        return self.base_dir.glob("*.pkl.gz")

    def load_episodes(self, episode_list):
        ret = []
        for episode_name in episode_list:
            path = self.base_dir / episode_name
            with gzip.open(path, mode="rb") as f:
                episode_data = pickle.load(f)
                ret.append(episode_data)  # format -> see TrainingDataRecorder#end_episode()
        return ret

    def forget_past(self):
        all_episodes = list(sorted(self.episodes(), reverse=True))
        if len(all_episodes) > self.config.train.memory_size:
            logger.info(f"forget old {len(all_episodes) - self.config.train.memory_size} episodes")
            for ep in all_episodes[self.config.train.memory_size:]:
                try:
                    Path(ep).unlink()
                except Exception as e:
                    logger.info(f"can not remove {ep}: {e}")
