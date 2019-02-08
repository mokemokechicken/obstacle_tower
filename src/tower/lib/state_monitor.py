from tower.agents.version1.state_model import StateModel
from tower.observation.event_handlers.base import EventHandler
from tower.observation.event_handlers.frame import FrameHistory
from tower.observation.event_handlers.infomation import InformationHandler


class StateMonitor(EventHandler):
    def __init__(self, state_model: StateModel, frame_history: FrameHistory, info: InformationHandler):
        self.state_model = state_model
        self.frame_history = frame_history
        self.info = info

    def before_step(self):
        frame = self.state_model.reconstruct_from_frame(self.frame_history.last_half_frame)
        self.info.screen.show("reconstruct", frame)
