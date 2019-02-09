from logging import getLogger

from obstacle_tower_env import ObstacleTowerEnv

from tower.agents.base import AgentBase
from tower.config import Config

logger = getLogger(__name__)


def start(config: Config, agent_cls):
    PlayCommand(config).start(agent_cls)


class PlayCommand:
    def __init__(self, config: Config):
        self.config = config

    def start(self, agent_cls, env_id=1):
        env = ObstacleTowerEnv(str(self.config.resource.obstacle_tower_path), retro=False, worker_id=env_id)

        agent: AgentBase = agent_cls(self.config, env)
        agent.setup()
        agent.play()
