from tower.agents.version1.policy_model import PolicyModel
from tower.agents.version1.state_model import StateModel
from tower.config import Config


class PolicyReTrainer:
    def __init__(self, config: Config, policy_model: PolicyModel, from_state: StateModel, to_state: StateModel):
        self.config = config
        self.policy_model = policy_model
        self.from_state = from_state
        self.to_state = to_state

    def fit(self):
        pass