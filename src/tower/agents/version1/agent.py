from logging import getLogger

from tower.agents.base import AgentBase
from tower.agents.version1.policy_model import PolicyModel
from tower.agents.version1.state_model import StateModel
from tower.const import Action
from tower.lib.memory import FileMemory
from tower.lib.state_monitor import StateMonitor
from tower.observation.event_handlers.infomation import InformationHandler
from tower.observation.event_handlers.training_data_recorder import TrainingDataRecorder
from tower.observation.manager import ObservationManager
import numpy as np

logger = getLogger(__name__)


class EvolutionAgent(AgentBase):
    state_model: StateModel
    policy_model: PolicyModel
    observation: ObservationManager

    def setup(self):
        self.observation = ObservationManager(self.config, self.env)
        self.observation.setup()
        self.state_model = StateModel(self.config)
        if not self.state_model.load_model():
            raise RuntimeError("No State Model Found")
        self.policy_model = PolicyModel(self.config)
        if not self.policy_model.load_model():
            self.policy_model.build()

        if self.config.play.render:
            info = InformationHandler(self.config, self.observation)
            self.observation.add_event_handler("info", info)
            state_monitor = StateMonitor(self.state_model, self.observation.frame_history, info)
            self.observation.add_event_handler("state_monitor", state_monitor)

        recorder = TrainingDataRecorder(self.config, FileMemory(self.config), self.observation)
        self.observation.add_event_handler("recorder", recorder)

    def play(self):
        n_episode = self.config.play.n_episode
        parameters = self.policy_model.get_parameters()
        logger.info([x.shape for x in parameters])  # (n_params, n_actions)
        for ep in range(n_episode):
            self.observation.floor(1)
            self.observation.reset()

            self.observation.begin_episode(ep)
            logger.info(f"start episode {ep}/{n_episode}")
            self.play_episode()
            logger.info(f"finish episode {ep}/{n_episode}")
            self.observation.end_episode(ep)

    def play_episode(self):
        done = False
        last_obs = None
        max_time_remain = 1

        while not done:
            self.observation.begin_loop()
            if last_obs is None:
                last_obs = [None, 0, 1]

            last_obs[0] = self.observation.frame_history.last_half_frame
            last_obs[1] = last_obs[1] / 5.
            last_obs[2] = 1.0 * last_obs[2] / max_time_remain

            action = self.decide_action(last_obs)
            obs, reward, done, info = self.observation.step(action)
            if reward != 0:
                logger.info(f"Get Reward={reward} Keys={obs[1]}")

            self.observation.end_loop()
            last_obs = list(obs)
            if max_time_remain < last_obs[2]:
                max_time_remain = last_obs[2]

    def decide_action(self, obs):
        state, sigma = self.state_model.encode_to_state(obs[0])
        actions = self.policy_model.predict(state, obs[1], obs[2])
        action = np.random.choice(range(len(actions)), p=actions)
        # action = np.argmax(actions)
        return LimitedAction.from_int(action)


class LimitedAction:
    size = 9
    actions = [
        Action.NOP,
        Action.FORWARD,
        Action.BACK,
        Action.LEFT,
        Action.RIGHT,
        Action.CAMERA_RIGHT,
        Action.CAMERA_LEFT,
        Action.JUMP,
        Action.FORWARD + Action.JUMP,
    ]

    @classmethod
    def from_int(cls, n):
        assert 0 <= n < cls.size
        return cls.actions[n]

