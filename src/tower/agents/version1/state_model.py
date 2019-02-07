from logging import getLogger

from tower.agents.version1.vae_model import VAEModel
from tower.config import Config

from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np
from tensorflow.python import keras


logger = getLogger(__name__)


class StateModel:
    def __init__(self, config: Config):
        self.config = config
        self.model: VAEModel = None

    def can_load(self):
        return False

    def load_model(self):
        raise NotImplemented()

    def build(self):
        mc = self.config.model
        self.model = VAEModel(self.config)
        self.model.build(mc.frame_shape)

    def compile(self):
        self.model.compile()

    def fit(self, frame: np.ndarray, next_frame: np.ndarray, action: np.ndarray):
        logger.info(f"fit data shape: frame={frame.shape}, next_frame={next_frame.shape}, action={action.shape}")

        trues = np.concatenate([frame, next_frame], axis=-1)
        self.model.training_model.fit([frame, action], trues, epochs=1)
