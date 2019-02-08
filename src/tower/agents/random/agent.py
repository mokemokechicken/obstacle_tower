from logging import getLogger

from tower.agents.base import AgentBase
from tower.agents.random.random_repeat_actor import RandomRepeatActor
from tower.agents.version1.state_model import StateModel
from tower.const import Action
from tower.lib.memory import FileMemory
from tower.lib.state_monitor import StateMonitor
from tower.observation.event_handlers.infomation import InformationHandler
from tower.observation.event_handlers.training_data_recorder import TrainingDataRecorder
from tower.observation.manager import ObservationManager

logger = getLogger(__name__)


class RandomAgent(AgentBase):
    observation: ObservationManager
    actor: RandomRepeatActor
    state_model: StateModel

    def setup(self):
        self.observation = ObservationManager(self.config, self.env)
        self.observation.setup()
        self.actor = self.create_random_actor()

        self.state_model = StateModel(self.config)
        self.state_model.load_model()

        if self.config.play.render:
            info = InformationHandler(self.config, self.observation)
            self.observation.add_event_handler("info", info)
            state_monitor = StateMonitor(self.state_model, self.observation.frame_history, info)
            self.observation.add_event_handler("state_monitor", state_monitor)

        recorder = TrainingDataRecorder(self.config, FileMemory(self.config), self.observation)
        self.observation.add_event_handler("recorder", recorder)

    @staticmethod
    def create_random_actor():
        random_actor = RandomRepeatActor(continue_rate=0.9)
        random_actor.reset(schedules=[
            (Action.CAMERA_RIGHT, 3),
            (Action.CAMERA_LEFT, 6),
            (Action.CAMERA_RIGHT, 3),
            (Action.NOP, 5),
            (Action.FORWARD, 8),
            (Action.RIGHT, 2),
            (Action.LEFT, 4),
            (Action.RIGHT, 2),
        ])
        return random_actor

    def play(self):
        n_episode = self.config.play.n_episode
        for ep in range(n_episode):
            self.observation.floor((ep % 25) + 1)
            self.observation.reset()

            self.observation.begin_episode(ep)
            logger.info(f"start episode {ep}/{n_episode}")
            self.play_episode()
            logger.info(f"finish episode {ep}/{n_episode}")
            self.observation.end_episode(ep)

    def play_episode(self):
        done = False
        while not done:
            self.observation.begin_loop()

            action = self.actor.decide_action(self.observation.moving_checker.did_move)
            obs, reward, done, info = self.observation.step(action)
            if reward != 0:
                logger.info(f"Get Reward={reward} Keys={obs[1]}")

            self.observation.end_loop()

