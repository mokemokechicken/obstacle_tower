import argparse
from logging import getLogger

from tower.agents.random.agent import RandomAgent
from tower.agents.version1.agent import EvolutionAgent
from tower.command import play, train
from tower.config import load_config
from tower.lib.logger import setup_logger

logger = getLogger(__name__)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help="command to run", choices=["train", "play"])
    parser.add_argument('-c', help="config file path", dest="config")
    parser.add_argument('--ep', help="number of episode play", type=int)
    parser.add_argument('--render', help="render screen when playing", action="store_true")
    parser.add_argument('--render-state', help="render state on screen when playing", action="store_true")
    parser.add_argument('--wait', type=int, help="wait milli seconds per frame (0 is forever)")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--new-model', action="store_true", help="create new state model")
    parser.add_argument('--random', action="store_true", help="play with RandomAgent")
    parser.add_argument('--no-save', action="store_true", help="play without recording")
    parser.add_argument('--no-update-state', action="store_true", help="not checking update of state model")
    return parser


def start():
    parser = create_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.ep:
        config.play.n_episode = args.ep
    if args.render:
        config.play.render = True
    if args.render_state:
        config.play.render_state = True
    if args.wait is not None:
        config.play.wait_per_frame = args.wait
    if args.debug:
        config.debug = True
    if args.new_model:
        config.train.new_model = True
    if args.no_save:
        config.play.no_save = True
    if args.no_update_state:
        config.evolution.no_update_state = True

    log_level = 'debug' if config.debug else 'info'
    setup_logger(config.resource.log_file_path, log_level)

    if args.command == "play":
        agent = RandomAgent if args.random else EvolutionAgent
        play.start(config, agent)
    elif args.command == "train":
        train.start(config)
