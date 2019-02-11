from collections import Counter
from logging import getLogger

import numpy as np
from tensorflow.python.keras.callbacks import Callback

from tower.agents.version1.limited_action import LimitedAction
from tower.agents.version1.policy_model import PolicyModel
from tower.agents.version1.state_model import StateModel
from tower.config import Config
from tower.lib.memory import FileMemory
from tower.lib.state_history import StateHistory

logger = getLogger(__name__)


class PolicyReTrainer:
    def __init__(self, config: Config, policy_model: PolicyModel, from_state: StateModel, to_state: StateModel):
        self.config = config
        self.policy_model = policy_model
        self.from_state = from_state
        self.to_state = to_state
        self.current_policy_model = None

    def train(self, memory: FileMemory):
        tc = self.config.policy_model_training
        self.current_policy_model = PolicyModel(self.config)
        self.current_policy_model.load_model()

        dx, dy = self.pickup_episodes(memory, tc.pickup_episodes)
        callbacks = [
            JustLoggingCallback(),
        ]
        self.policy_model.compile()
        self.policy_model.model.fit(dx, dy, batch_size=tc.batch_size, epochs=tc.epochs, callbacks=callbacks)

    def pickup_episodes(self, memory: FileMemory, size=None):
        all_episodes = list(memory.episodes())
        logger.info(f"{len(all_episodes)} episodes found")
        if size:
            all_episodes = self.pickup_top_n_rewards_episodes(memory, all_episodes, size)
            logger.info(f"best {size} episodes are picked up")

        input_list = []
        output_list = []
        for ei, ep in enumerate(all_episodes):
            logger.info(f"loading {ei+1}/{len(all_episodes)} episode")
            episode_data = memory.load_episodes([ep])
            if not episode_data:
                continue
            input_data, output_data = self.create_dataset(episode_data[0])
            input_list += input_data
            output_list += output_data
        data_x = [np.array([x[i] for x in input_list]) for i in range(5)]
        data_y = [np.array([x[i] for x in output_list]) for i in range(2)]
        return data_x, data_y

    def pickup_top_n_rewards_episodes(self, memory, all_episodes, size):
        episodes = Counter()
        for name in all_episodes:
            ep_data = memory.load_episodes([name])
            if not ep_data:
                continue
            meta = ep_data[0].get("meta", {})
            reward = meta.get("reward", 0)
            map_reward = meta.get("map_reward", 0)
            episodes[name] = reward + map_reward * self.config.train.map_reward_weight
        return [x[0] for x in episodes.most_common(size)]

    def create_dataset(self, episode_data):
        steps = episode_data["steps"]
        max_time_remain = float(max([x["state"][2] for x in steps]))
        action_history = []

        input_data = []
        output_data = []
        cur_state_history = StateHistory(self.config.model.vae.latent_dim + 1)
        new_state_history = StateHistory(self.config.model.vae.latent_dim + 1)

        for step in steps:
            obs = step["state"]
            recorded_action = LimitedAction.original_action_to_limited_action(step["action"])
            obs[0] = obs[0] / 255.
            obs[1] /= 5.
            obs[2] /= float(max_time_remain)

            in_actions = np.zeros((self.config.policy_model.n_actions,))
            for past_action in action_history:
                in_actions[past_action] += 1
            in_actions /= self.config.evolution.action_history_size
            action_history.append(recorded_action)
            action_history = action_history[-self.config.evolution.action_history_size:]

            state, _ = self.from_state.encode_to_state(obs[0])
            new_state, _ = self.to_state.encode_to_state(obs[0])

            cur_state_history.store(state, obs)
            new_state_history.store(new_state, obs)
            cur_rarity = cur_state_history.recent_rarity * self.config.policy_model.recent_rarity_weight
            new_rarity = new_state_history.recent_rarity * self.config.policy_model.recent_rarity_weight

            actions, keep_rate = self.current_policy_model.predict(state, obs[1], obs[2], in_actions, cur_rarity)

            input_data.append((new_state, [obs[1]], [obs[2]], in_actions, [new_rarity]))
            output_data.append((actions, keep_rate))

        return input_data, output_data


class JustLoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logger.info(f"epoch {epoch} logs {logs}")
