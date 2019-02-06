from ..base import TrainerBase
from .a2c_model import AgentModel


class Trainer(TrainerBase):
    model: AgentModel

    def train(self):
        self.model = self.load_model()

    def load_model(self):
        model = AgentModel(self.config)
        if self.config.train.new_model or not model.can_load():
            model.build()
        else:
            model.load_model()
        return model


