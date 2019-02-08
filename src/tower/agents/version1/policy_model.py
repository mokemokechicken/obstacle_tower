from logging import getLogger

import numpy as np
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.models import load_model
from tensorflow.python.layers.core import Dense

from tower.config import Config

logger = getLogger(__name__)


class PolicyModel:
    def __init__(self, config: Config):
        self.config = config
        self.model: Model = None
        self.parameters: Dense = None

    def load_model(self):
        try:
            self.model = load_model(str(self.model_file_path))
            logger.info(f"loading policy model success")
            return True
        except Exception:
            return False

    def save_model(self):
        logger.info(f"saving policy model")
        self.model_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.model_file_path), include_optimizer=False)

    def build(self):
        logger.info(f"setup state model")
        in_state = Input((self.config.model.vae.latent_dim, ), name="in_state")
        in_keys = Input((1, ), name="in_keys")
        in_time = Input((1, ), name="in_time")
        in_all = Concatenate(name="in_all")([in_state, in_keys, in_time])
        self.parameters = Dense(self.config.policy_model.n_actions, activation="softmax", name="parameters")
        out_actions = self.parameters(in_all)
        self.model = Model([in_state, in_keys, in_time], out_actions, name="policy_model")

    def predict(self, state, keys, time_remain):
        actions = self.model.predict([np.expand_dims(state, axis=0), np.array([[keys]]), np.array([[time_remain]])])[0]
        return actions

    def get_parameters(self):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def compile(self):
        self.model.compile()

    @property
    def model_file_path(self):
        return self.config.resource.model_dir / "policy_weights.h5"


