from tower.agents.version1.limited_action import LimitedAction
from tower.agents.version1.policy_model import PolicyModel
from tower.agents.version1.state_model import StateModel
from tower.config import Config
from tower.lib.memory import FileMemory
import numpy as np



class PolicyReTrainer:
    def __init__(self, config: Config, policy_model: PolicyModel, from_state: StateModel, to_state: StateModel):
        self.config = config
        self.policy_model = policy_model
        self.from_state = from_state
        self.to_state = to_state

    def train(self, memory: FileMemory):
        tc = self.config.policy_model_config
        self.policy_model.compile()
        data_generator = self.generator(memory)
        self.policy_model.model.fit_generator(data_generator, steps_per_epoch=tc.steps_per_epoch, epochs=tc.epochs)

    def generator(self, memory: FileMemory):
        all_episodes = memory.episodes()
        for ep in all_episodes:
            input_data, output_data = self.create_dataset(memory.load_episodes([ep])[0])
            data_x = [[np.array(x[0]), np.array(x[1]), np.array(x[2]), np.array(x[3])] for x in input_data]
            data_y = [[np.array(x[0]), np.array(x[1])] for x in output_data]
            yield (data_x, data_y)

    def create_dataset(self, episode_data):
        steps = episode_data["steps"]
        max_time_remain = float(max([x["state"][2] for x in steps]))
        action_history = []

        input_data = []
        output_data = []

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
            actions, keep_rate = self.policy_model.predict(state, obs[1], obs[2], in_actions)

            input_data.append((new_state, [obs[1]], [obs[2]], in_actions))
            output_data.append((actions, [keep_rate]))

        return input_data, output_data


