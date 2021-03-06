from logging import getLogger

import numpy as np
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.losses import kullback_leibler_divergence, mean_squared_error
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.layers.core import Dense

from tower.config import Config

logger = getLogger(__name__)


class PolicyModel:
    def __init__(self, config: Config):
        self.config = config
        self.model: Model = None

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
        in_state = Input((self.config.model.vae.latent_dim,), name="in_state")
        in_keys = Input((1,), name="in_keys")
        in_time = Input((1,), name="in_time")
        in_actions = Input((self.config.policy_model.n_actions,), name="in_actions")
        in_rarity = Input((1,), name="in_rarity")
        in_all = Concatenate(name="in_all")([in_state, in_keys, in_time, in_actions, in_rarity])
        x = Dense(self.config.policy_model.hidden_size, activation="tanh", name="hidden",
                  kernel_regularizer=l2(0.0001))(in_all)
        out_actions = Dense(self.config.policy_model.n_actions, activation="softmax", name="parameters",
                            kernel_regularizer=l2(0.0001))(x)
        out_keep_rate = Dense(1, activation="sigmoid", name="keep_rate")(in_all)
        self.model = Model([in_state, in_keys, in_time, in_actions, in_rarity], [out_actions, out_keep_rate],
                           name="policy_model")

    def predict(self, state, keys, time_remain, in_actions, in_rarity):
        actions, kr = self.model.predict([
            np.expand_dims(state, axis=0),
            np.array([[keys]]),
            np.array([[time_remain]]),
            np.expand_dims(in_actions, axis=0),
            np.array([[in_rarity]]),
        ])
        return actions[0], kr[0]

    def get_parameters(self):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def compile(self):
        self.model.compile(optimizer=Adam(lr=0.00001), loss=[kullback_leibler_divergence, mean_squared_error],
                           loss_weights=[1., 0.])

    @property
    def model_file_path(self):
        return self.config.resource.model_dir / "policy_weights.h5"
