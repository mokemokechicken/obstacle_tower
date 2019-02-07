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
        return self.encoder_file_path.exists() and self.decoder_file_path.exists()

    def load_model(self):
        self.build()
        logger.info(f"loading state model")
        self.model.encoder.load_weights(str(self.encoder_file_path))
        self.model.decoder.load_weights(str(self.decoder_file_path))

    def save_model(self):
        logger.info(f"saving state model")
        self.encoder_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.encoder.save_weights(str(self.encoder_file_path))
        self.model.decoder.save_weights(str(self.decoder_file_path))

    def build(self):
        logger.info(f"setup state model")
        mc = self.config.model
        self.model = VAEModel(self.config)
        self.model.build(mc.frame_shape)

    def compile(self):
        self.model.compile()

    @property
    def encoder_file_path(self):
        return self.config.resource.model_dir / "state_encoder_weights.hf5"

    @property
    def decoder_file_path(self):
        return self.config.resource.model_dir / "state_decoder_weights.hf5"


