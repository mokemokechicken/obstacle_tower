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


class ResourceConfig(ConfigBase):
    def __init__(self):
        self.obstacle_tower_path = _project_base_dir() / "obstacletower"
        self.log_file_path = _project_base_dir() / "log" / "tower.log"
        self.working_dir = _project_base_dir() / "tmp" / "working"
        self.memory_dir = _project_base_dir() / "data" / "memory"
        self.model_dir = _project_base_dir() / "data" / "model"


class PlayConfig(ConfigBase):
    def __init__(self):
        self.render = False
        self.wait_per_frame = 1
        self.n_episode = 1


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
        self.importance_step = 10
        self.map_reward_weight = 0.0001
        self.max_episode_in_one_time = 1


class VAETrainConfig(ConfigBase):
    def __init__(self):
        self.kl_loss_rate = 0.1
        self.next_state_loss_weight = 0.1
        self.lr = 0.01
        self.lr_decay_factor = 0.1
        self.lr_patience = 10
        self.lr_min = 0.00001
        self.batch_size = 512
        self.epochs = 100


class ModelConfig(ConfigBase):
    def __init__(self):
        self.frame_shape = (168//2, 168//2, 3)
        self.max_key_num = 5
        self.vae = VAEModelConfig()


class VAEModelConfig(ConfigBase):
    def __init__(self):
        self.conv_layers = [
            dict(filters=32, kernel_size=8, strides=4, activation="relu"),
            dict(filters=32, kernel_size=3, strides=1, activation="relu"),
        ]
        self.latent_dim = 8
        self.action_size = 54
