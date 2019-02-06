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


class ResourceConfig(ConfigBase):
    def __init__(self):
        self.obstacle_tower_path = _project_base_dir() / "obstacletower"
        self.log_file_path = _project_base_dir() / "log" / "tower.log"
        self.working_dir = _project_base_dir() / "tmp" / "working"
        self.memory_dir = _project_base_dir() / "data" / "memory"


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
