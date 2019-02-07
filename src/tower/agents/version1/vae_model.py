from logging import getLogger

import numpy as np
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.layers import Dense, Lambda, Reshape, Conv2D, Conv2DTranspose, Flatten, Concatenate
from tensorflow.python.keras.optimizers import Adam

from tower.config import Config

logger = getLogger(__name__)


class VAEModel:
    def __init__(self, config: Config):
        self.config = config
        self.encoder = None  # type: Model
        self.decoder = None  # type: Model
        self.z_mean = None
        self.z_log_var = None
        self.training_model = None  # type: Model
        self.next_z_mean = None

    def build(self, feature_shape):
        vc = self.config.model.vae

        # Encoder
        frame_in = Input(feature_shape, name="VAE/x_input")
        hidden = frame_in
        for i, conv in enumerate(vc.conv_layers):
            hidden = Conv2D(name=f"VAE/encoder_conv2D_{i + 1}", **conv)(hidden)
        encoder_last_shape = tuple(x for x in K.int_shape(hidden) if x is not None)
        encoder_last_layer = hidden = Flatten()(hidden)
        self.z_mean = Dense(vc.latent_dim, activation='linear', name="VAE/latent_mean")(hidden)
        self.z_log_var = Dense(vc.latent_dim, activation='linear', name="VAE/latent_var")(hidden)
        z = Lambda(self.sampling, output_shape=(vc.latent_dim,), name="VAE/sampling")([self.z_mean, self.z_log_var])

        # Decoder
        z_placeholder = Input((vc.latent_dim,), name="VAE/z_input")
        h_decoded = Dense(np.prod(encoder_last_shape), activation='relu', name="VAE/decode_fc0")(z_placeholder)
        h_decoded = Reshape(encoder_last_shape)(h_decoded)
        for i, conv in enumerate(reversed(vc.conv_layers)):
            h_decoded = Conv2DTranspose(name=f"VAE/decoder_conv2D_{i + 1}", **conv)(h_decoded)
        h_decoded = Conv2D(filters=3, kernel_size=1, strides=1, activation="linear")(h_decoded)
        assert feature_shape == K.int_shape(h_decoded)[1:], f"{feature_shape} != {K.int_shape(h_decoded)[1:]}"
        x_decoded_mean = h_decoded

        # next_latent_z
        action_in = Input((vc.action_size,))
        encoder_and_action = Concatenate()([encoder_last_layer, action_in])
        self.next_z_mean = Dense(vc.latent_dim, activation='linear', name="VAE/next_latent_mean")(encoder_and_action)

        self.encoder = Model(frame_in, self.z_mean, name="VAE/encoder")
        self.decoder = Model(z_placeholder, x_decoded_mean, name="VAE/decoder")
        self.training_model = Model([frame_in, action_in], self.decoder(z), name="VAE/training")

    def compile(self):
        self.training_model.compile(optimizer=Adam(lr=self.config.train.vae.lr), loss=self.vae_loss)

    @staticmethod
    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def vae_loss(self, trues, x_decoded_mean):
        z_mean, z_log_var, next_z_mean = self.z_mean, self.z_log_var, self.next_z_mean
        current_frame, next_frame = trues

        # 次フレーム
        next_state_loss = K.sum(K.square(self.encoder(next_frame) - next_z_mean), axis=-1)

        # 1項目の計算
        latent_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        # 2項目の計算
        reconstruct_loss = self.reconstruct_loss(current_frame, x_decoded_mean)
        return K.mean(latent_loss * self.config.train.vae.kl_loss_rate + reconstruct_loss +
                      next_state_loss * self.config.train.vae.next_state_loss_weight)

    @staticmethod
    def reconstruct_loss(y_true, y_pred):
        return K.sum(K.square(y_pred - y_true), axis=-1)

    def freeze(self):
        for l in self.training_model.layers:
            l.trainable = False


def train_auto_encoder(config: Config, vae_model, data_x):
    """

    :param Config config:
    :param VAEModel vae_model:
    :param np.ndarray data_x:
    :param bool freeze_after_training:
    :return:
    """
    cf = config.train.vae
    vae_model.compile()
    callbacks = [
        ReduceLROnPlateau(factor=cf.lr_decay_factor, patience=cf.lr_patience, min_lr=cf.lr_min, monitor='loss',
                          verbose=1),
    ]

    vae_model.training_model.fit(data_x, data_x, batch_size=cf.batch_size, epochs=cf.epochs, verbose=2,
                                 callbacks=callbacks)

    z = vae_model.encoder.predict(data_x)
    logger.info(f"VAE Latent: mean={np.mean(z, axis=0)}, cov={np.cov(z.T)}")
