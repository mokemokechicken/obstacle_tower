from logging import getLogger

from obstacle_tower_env import ObstacleTowerEnv

from tower.agents.base import AgentBase
from tower.config import Config
from tower.lib.screen import Screen

logger = getLogger(__name__)


def start(config: Config, agent_cls):
    PlayCommand(config).start(agent_cls)


class PlayCommand:
    def __init__(self, config: Config):
        self.config = config
        self.screen: Screen = None

    def start(self, agent_cls):
        env = ObstacleTowerEnv(str(self.config.resource.obstacle_tower_path), retro=False, worker_id=9)
        agent: AgentBase = agent_cls(self.config, env)
        agent.setup()
        agent.play()
