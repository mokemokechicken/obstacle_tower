import argparse
from logging import getLogger

from tower.command import play
from tower.config import load_config
from tower.lib.logger import setup_logger

logger = getLogger(__name__)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', help="config file path", dest="config")
    parser.add_argument('--render', help="render screen when playing", action="store_true")
    parser.add_argument('--wait', type=int, help="wait milli seconds per frame (0 is forever)")
    return parser


def start():
    parser = create_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logger(config.resource.log_file_path, 'info')

    if args.render:
        config.play.render = args.render
    if args.wait is not None:
        config.play.wait_per_frame = args.wait

    play.start(config)
