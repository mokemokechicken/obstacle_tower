from pathlib import Path

import yaml
from moke_config import ConfigBase, create_config


def _project_base_dir() -> Path:
    return Path(__file__).absolute().parents[2]


def load_config(config_path=None):
    """

    :param config_path:
    :rtype: Config
    :return:
    """
    config_dict = {}
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with config_path.open("rt") as f:
                config_dict = yaml.load(f)
    return create_config(Config, config_dict)


class Config(ConfigBase):
    def __init__(self):
        self.debug = False
        self.resource = ResourceConfig()
        self.play = PlayConfig()
        self.map = MapConfig()
        self.train = TrainConfig()
        self.model = ModelConfig()
        self.evolution = EvolutionConfig()
        self.policy_model = PolicyModelConfig()
        self.policy_model_training = PolicyModelTrainingConfig()


class ResourceConfig(ConfigBase):
    def __init__(self):
        self.obstacle_tower_path = _project_base_dir() / "obstacletower"
        self.log_file_path = _project_base_dir() / "log" / "tower.log"
        self.working_dir = _project_base_dir() / "tmp" / "working"
        self.memory_dir = _project_base_dir() / "data" / "memory"
        self.model_dir = _project_base_dir() / "data" / "model"
        self.new_model_dir = _project_base_dir() / "data" / "new_model"
        self.state_db_dir = _project_base_dir() / "data" / "state_db"


class PlayConfig(ConfigBase):
    def __init__(self):
        self.render = False
        self.wait_per_frame = 1
        self.n_episode = 1
        self.render_state = False
        self.no_save = False


class MapConfig(ConfigBase):
    def __init__(self):
        self.map_size = 64
        self.visit_map_scale = 5.
        self.visit_map_value = 0.2
        self.wall_map_scale = 1.
        self.wall_map_value = 0.3


class TrainConfig(ConfigBase):
    def __init__(self):
        self.new_model = False
        self.vae = VAETrainConfig()
        self.importance_step = 20
        self.importance_scale = 1.
        self.map_reward_weight = 0.000001
        self.death_reward_weight = 0.1
        self.max_episode_in_one_time = 100
        self.discount_rate = 0.97
        self.memory_size = 1000


class VAETrainConfig(ConfigBase):
    def __init__(self):
        self.reconstruct_loss_weight = 0.01
        self.kl_loss_rate = 0.1
        self.next_state_loss_weight = 1.
        self.lr = 0.0001
        self.lr_decay_factor = 0.1
        self.lr_patience = 30
        self.lr_min = self.lr * 0.001
        self.epochs = 100
        self.steps_per_epoch = 1000
        self.batch_size = 32
        self.early_stopping_patience = 50


class ModelConfig(ConfigBase):
    def __init__(self):
        self.frame_shape = (168//2, 168//2, 3)
        self.max_key_num = 5
        self.vae = VAEModelConfig()


class VAEModelConfig(ConfigBase):
    def __init__(self):
        self.hsv_model = False
        self.conv_layers = [
            # dict(filters=32, kernel_size=4, strides=2, activation="relu", padding="same"),
            # dict(filters=32, kernel_size=3, strides=2, activation="relu", padding="same"),
            # dict(filters=128, kernel_size=4, strides=2, activation="relu", padding="same"),
            # dict(filters=256, kernel_size=3, strides=2, activation="relu", padding="same"),
            # dict(filters=512, kernel_size=3, strides=1, activation="relu", padding="same"),
            dict(filters=64, kernel_size=3, strides=3, activation="relu", padding="same"),
            dict(filters=64, kernel_size=3, strides=2, activation="relu", padding="same"),
            dict(filters=64, kernel_size=3, strides=2, activation="relu", padding="same"),
        ]
        self.latent_dim = 8  # 24  # 8
        self.action_size = 54


class EvolutionConfig(ConfigBase):
    def __init__(self):
        self.n_epoch = 25
        self.n_play_per_test = 3
        self.n_test_per_epoch = 7
        self.learning_rate = 0.1
        self.noise_sigma = 0.5
        self.use_best_action = False
        self.action_history_size = 10
        self.no_update_state = False
        self.start_random_floor = False
        self.explore_reward_weight = 0.01


class PolicyModelConfig(ConfigBase):
    def __init__(self):
        self.n_actions = 9
        self.hidden_size = 8
        self.recent_rarity_weight = 20.


class PolicyModelTrainingConfig(ConfigBase):
    def __init__(self):
        self.pickup_episodes = 1000
        self.epochs = 25
        self.batch_size = 32
