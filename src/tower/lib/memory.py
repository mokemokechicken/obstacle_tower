import pickle
from datetime import datetime
from uuid import uuid4

from tower.config import Config


class Memory:
    def setup(self):
        pass

    def store(self, data):
        pass


class FileMemory(Memory):
    def __init__(self, config: Config):
        self.config = config
        self.base_dir = self.config.resource.memory_dir

    def setup(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def store(self, data):
        dt = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = self.base_dir / f"{dt}_{uuid4().hex}.pkl"
        with path.open(mode="wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)