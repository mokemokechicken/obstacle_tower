import shutil
from logging import getLogger

import numpy as np

from tower.agents.base import AgentBase
from tower.agents.version1.limited_action import LimitedAction
from tower.agents.version1.policy_model import PolicyModel
from tower.agents.version1.state_model import StateModel
from tower.agents.version1.train_policy_in_another_state import PolicyReTrainer
from tower.lib.memory import FileMemory
from tower.lib.state_monitor import StateMonitor
from tower.observation.event_handlers.infomation import InformationHandler
from tower.observation.event_handlers.training_data_recorder import TrainingDataRecorder
from tower.observation.manager import ObservationManager

logger = getLogger(__name__)


class EvolutionAgent(AgentBase):
    state_model: StateModel
    policy_model: PolicyModel
    observation: ObservationManager
    start_floor: int
    action_history: list

    def setup(self):
        self.observation = ObservationManager(self.config, self.env)
        self.observation.setup()
        self.state_model = StateModel(self.config)
        if not self.state_model.load_model():
            logger.info(f"No State Model Found")
            self.state_model.build()
            self.state_model.save_model()
        self.policy_model = PolicyModel(self.config)
        if not self.policy_model.load_model():
            self.policy_model.build()

        if self.config.play.render:
            info = InformationHandler(self.config, self.observation)
            self.observation.add_event_handler("info", info)
            if self.config.play.render_state:
                state_monitor = StateMonitor(self.state_model, self.observation.frame_history, info)
                self.observation.add_event_handler("state_monitor", state_monitor)

        if not self.config.play.no_save:
            recorder = TrainingDataRecorder(self.config, FileMemory(self.config), self.observation)
            self.observation.add_event_handler("recorder", recorder)

    def play(self):
        ec = self.config.evolution
        best_rewards = []

        for epoch_idx in range(ec.n_epoch):
            self.check_and_update_of_state_model()
            logger.info(f"Start Training Epoch: {epoch_idx+1}/{ec.n_epoch}")
            self.start_floor = (epoch_idx % 20) + 1
            test_results = []
            original_parameters = self.policy_model.get_parameters()
            for test_idx in range(ec.n_test_per_epoch):
                logger.info(f"Start Test: Epoch={epoch_idx+1} test={test_idx+1}/{ec.n_test_per_epoch}")
                new_parameters, noises = self.make_new_parameters(original_parameters, sigma=ec.noise_sigma)
                self.policy_model.set_parameters(new_parameters)
                reward = self.play_n_episode(ec.n_play_per_test)
                logger.info(f"Finish Test: Epoch={epoch_idx+1} test={test_idx+1}/{ec.n_test_per_epoch} -> mean reward={reward}")
                test_results.append((reward, noises))
            new_parameters = self.update_parameters(original_parameters, test_results)
            self.policy_model.set_parameters(new_parameters)
            logger.info(f"Finish Training Epoch: {epoch_idx+1}/{ec.n_epoch}")
            self.policy_model.save_model()
            best_rewards.append(float(np.max([x[0] for x in test_results])))
            logger.info(f"best reward history={best_rewards}")

    def check_and_update_of_state_model(self):
        if not self.state_model.new_model_is_found():
            return
        new_state_model = StateModel(self.config)
        if not new_state_model.load_model(new_model=True):
            logger.warning(f"---------- loading new model fail!! -------------")
            return
        trainer = PolicyReTrainer(self.config, self.policy_model, self.state_model, new_state_model)
        trainer.train(FileMemory(self.config))

        self.state_model = new_state_model
        files = list(self.config.resource.new_model_dir.glob("state_*"))
        for f in files:
            logger.info(f"copying state model file: {f.name}")
            shutil.copy(f, self.config.resource.model_dir)

    def make_new_parameters(self, original_parameters, sigma):
        new_parameters = []
        noises = []
        for w in original_parameters:
            noise = np.random.randn(*w.shape)
            new_parameters.append(w + sigma * noise)
            noises.append(noise)
        return new_parameters, noises

    def update_parameters(self, original_parameters, test_results):
        ec = self.config.evolution
        rewards = np.array([x[0] for x in test_results])
        noises = np.array([x[1] for x in test_results])
        norm_rewards = (rewards - rewards.mean()) / (rewards.std() + 0.0000001)
        new_parameters = []
        for i, w in enumerate(original_parameters):
            noise_at_i = np.array([n[i] for n in noises])
            rate = ec.learning_rate / (len(test_results) * ec.noise_sigma)
            w = w + rate * np.dot(noise_at_i.T, norm_rewards).T
            new_parameters.append(w)
        return new_parameters

    def play_n_episode(self, n_episode) -> float:
        rewards = []
        for ep in range(n_episode):
            self.observation.floor(self.start_floor)
            self.observation.reset()

            self.observation.begin_episode(ep)
            logger.info(f"start episode {ep}/{n_episode}")
            rewards.append(self.play_episode())
            logger.info(f"finish episode {ep}/{n_episode}")
            self.observation.end_episode(ep)
        return float(np.mean(rewards))

    def play_episode(self) -> float:
        done = False
        last_obs = None
        max_time_remain = 1
        real_reward = 0
        map_reward = 0
        keep_rate = 0
        last_action = None
        self.action_history = []

        while not done:
            self.observation.begin_loop()
            if last_obs is None:
                last_obs = [None, 0, 1]

            last_obs[0] = self.observation.frame_history.last_half_frame
            last_obs[1] = last_obs[1] / 5.
            last_obs[2] = 1.0 * last_obs[2] / max_time_remain

            if last_action is None or keep_rate <= np.random.random() or not self.observation.moving_checker.did_move:
                action, keep_rate = self.decide_action(last_obs)
            else:
                action = last_action
            obs, reward, done, info = self.observation.step(action)
            last_action = action
            if reward != 0:
                logger.info(f"Get Reward={reward} Keys={obs[1]}")
            real_reward += reward
            map_reward += self.observation.map_observation.map_reward * self.config.train.map_reward_weight

            self.observation.end_loop()
            last_obs = list(obs)
            if max_time_remain < last_obs[2]:
                max_time_remain = last_obs[2]

        return real_reward + map_reward

    def decide_action(self, obs):
        state, sigma = self.state_model.encode_to_state(obs[0])

        in_actions = np.zeros((self.config.policy_model.n_actions, ))
        for past_action in self.action_history:
            in_actions[past_action] += 1
        in_actions /= self.config.evolution.action_history_size

        actions, keep_rate = self.policy_model.predict(state, obs[1], obs[2], in_actions)
        if self.config.evolution.use_best_action:
            action = np.argmax(actions)
        else:
            action = np.random.choice(range(len(actions)), p=actions)

        self.action_history.append(action)
        self.action_history = self.action_history[-self.config.evolution.action_history_size:]

        return LimitedAction.from_int(action), keep_rate * 0.9


