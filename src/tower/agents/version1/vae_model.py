from logging import getLogger

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.layers import Dense, Lambda, Reshape, Conv2D, Conv2DTranspose, Flatten, Concatenate, Input
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam, SGD

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
        self.next_z_log_var = None
        self.frame_in = None
        self.action_in = None

    def build(self, feature_shape):
        vc = self.config.model.vae

        # Encoder
        self.frame_in = Input(feature_shape, name="VAE/x_input")
        hidden = self.frame_in
        for i, conv in enumerate(vc.conv_layers):
            logger.info(f"conv2d param: {conv}")
            hidden = Conv2D(name=f"VAE/encoder_conv2D_{i + 1}", **conv)(hidden)
        encoder_last_shape = tuple(x for x in K.int_shape(hidden) if x is not None)
        logger.info(f"encoder_last_shape={encoder_last_shape}")
        hidden = Flatten()(hidden)
        self.z_mean = Dense(vc.latent_dim, activation='linear', name="VAE/latent_mean")(hidden)
        self.z_log_var = Dense(vc.latent_dim, activation='linear', name="VAE/latent_log_var")(hidden)
        z = Lambda(self.sampling, output_shape=(vc.latent_dim,), name="VAE/sampling")([self.z_mean, self.z_log_var])

        # Decoder
        z_placeholder = Input((vc.latent_dim,), name="VAE/z_input")
        h_decoded = Dense(np.prod(encoder_last_shape), activation='relu', name="VAE/decode_fc0")(z_placeholder)
        h_decoded = Reshape(encoder_last_shape)(h_decoded)
        for i, conv in enumerate(reversed(vc.conv_layers)):
            h_decoded = Conv2DTranspose(name=f"VAE/decoder_conv2D_{i + 1}", **conv)(h_decoded)

        if not vc.hsv_model:  # RGB Version
            h_decoded = Conv2D(filters=3, kernel_size=1, strides=1, activation="sigmoid")(h_decoded)
        else:  # HSV Version
            ch_h_decoded = Conv2D(filters=1, kernel_size=1, strides=1, activation="linear")(h_decoded)
            ch_h_decoded = Lambda(lambda x: x - K.round(x - 0.5))(ch_h_decoded)
            ch_sv_decoded = Conv2D(filters=2, kernel_size=1, strides=1, activation="sigmoid")(h_decoded)
            h_decoded = Concatenate(axis=-1)([ch_h_decoded, ch_sv_decoded])

        assert feature_shape == K.int_shape(h_decoded)[1:], f"{feature_shape} != {K.int_shape(h_decoded)[1:]}"
        x_decoded_mean = h_decoded

        # next_latent_z
        self.action_in = Input((vc.action_size,))
        state_and_action = Concatenate()([self.z_mean, self.z_log_var, self.action_in])
        self.next_z_mean = Dense(vc.latent_dim, activation='linear', name="VAE/next_latent_mean")(state_and_action)
        self.next_z_log_var = Dense(vc.latent_dim, activation='linear', name="VAE/next_latent_log_var")(
            state_and_action)

        # reward
        reward = Dense(1, activation="tanh", name="reward")(state_and_action)

        self.encoder = Model(self.frame_in, [self.z_mean, self.z_log_var], name="VAE/encoder")
        self.decoder = Model(z_placeholder, x_decoded_mean, name="VAE/decoder")
        self.training_model = Model([self.frame_in, self.action_in], [self.decoder(z), reward], name="VAE/training")

    def compile(self):
        optimizer = Adam(lr=self.config.train.vae.lr)
        # optimizer = SGD(lr=self.config.train.vae.lr, momentum=0.9)
        self.training_model.compile(optimizer=optimizer, loss=[self.vae_loss, mean_squared_error])

    @staticmethod
    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def vae_loss(self, targets, x_decoded_mean):
        z_mean, z_log_var, next_z_mean, next_z_log_var = self.z_mean, self.z_log_var, self.next_z_mean, self.next_z_log_var
        current_frame, next_frame = targets[:, :, :, 0:3], targets[:, :, :, 3:6]

        # 次フレーム
        next_z_mean_of_current, next_z_log_var_of_current = self.encoder(next_frame)
        # next_state_loss = K.sum(K.square(next_z_mean_of_current - next_z_mean), axis=-1)
        # KLD(next || current)
        next_state_loss = -0.5 * K.sum(1 + next_z_log_var - next_z_log_var_of_current -
                                       (K.exp(next_z_log_var) +
                                        K.square(next_z_mean - next_z_log_var_of_current)) / K.exp(next_z_log_var_of_current)
                                       , axis=-1)

        # 1項目の計算
        latent_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        # 2項目の計算
        if self.config.model.vae.hsv_model:
            reconstruct_loss = self.reconstruct_loss_hsv_image(current_frame, x_decoded_mean)
        else:
            reconstruct_loss = self.reconstruct_loss(current_frame, x_decoded_mean)

        cf = self.config.train.vae
        total_loss = reconstruct_loss * cf.reconstruct_loss_weight + latent_loss * cf.kl_loss_rate
        total_loss += next_state_loss * cf.next_state_loss_weight
        return K.mean(total_loss)

    @staticmethod
    def reconstruct_loss(y_true, y_pred):
        x = K.mean(K.square(y_pred - y_true), axis=[3])
        return K.sum(x, axis=[1, 2])

    @staticmethod
    def reconstruct_loss_hsv_image(y_true, y_pred):
        # dim means (batch, y, x, ch(hsv))
        v1 = K.square(y_true[:, :, :, 0:1] - y_pred[:, :, :, 0:1])
        v2 = K.square(y_true[:, :, :, 0:1] - (y_pred[:, :, :, 0:1] - 1))
        v3 = K.square(y_true[:, :, :, 0:1] - (y_pred[:, :, :, 0:1] + 1))
        vv = K.concatenate([v1, v2, v3], axis=3)  # H of HSV is cyclic around 0 to 1
        h_loss = K.min(vv, axis=3)
        sv_loss = K.sum(K.square(y_true[:, :, :, 1:3] - y_pred[:, :, :, 1:3]), axis=3)

        x = (h_loss + sv_loss/8.)  # sv_loss is less importance than h_loss
        return K.sum(x, axis=[1, 2])

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
