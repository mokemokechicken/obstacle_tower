import math
from logging import getLogger

from tower.agents.version1.vae_model import VAEModel
from tower.config import Config

from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np
from tensorflow.python import keras

from tower.lib.image_util import bgr_to_hsv, hsv_to_bgr

logger = getLogger(__name__)


class StateModel:
    def __init__(self, config: Config):
        self.config = config
        self.model: VAEModel = None

    def load_model(self):
        self.build()
        try:
            self.model.encoder.load_weights(str(self.encoder_file_path))
            self.model.decoder.load_weights(str(self.decoder_file_path))
            logger.info(f"loading state model success")
            return True
        except Exception:
            return False

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

    def encode_to_state(self, half_frame):
        if self.config.model.vae.hsv_model:
            half_frame = bgr_to_hsv(half_frame, to_float=True, from_float=True)
        z_means, z_log_vars = self.model.encoder.predict(np.expand_dims(half_frame, axis=0))
        z_sigma = np.sqrt(np.exp(z_log_vars[0]))
        return z_means[0], z_sigma

    def decode_from_state(self, state):
        frame = self.model.decoder.predict(np.expand_dims(state, axis=0))[0]
        if self.config.model.vae.hsv_model:
            frame = hsv_to_bgr(frame, to_float=True, from_float=True)
        return frame

    def reconstruct_from_frame(self, half_frame):
        z_mean, z_sigma = self.encode_to_state(half_frame)
        frame = self.decode_from_state(z_mean)
        return frame

    def compile(self):
        self.model.compile()

    @property
    def encoder_file_path(self):
        return self.config.resource.model_dir / "state_encoder_weights.hf5"

    @property
    def decoder_file_path(self):
        return self.config.resource.model_dir / "state_decoder_weights.hf5"


