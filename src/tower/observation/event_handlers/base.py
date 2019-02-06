from collections import namedtuple

EventParamsAfterStep = namedtuple('EventParamsAfterStep', 'action obs reward done info')


class EventHandler:
    def reset(self):
        pass

    def begin_loop(self):
        pass

    def before_step(self):
        pass

    def after_step(self, params: EventParamsAfterStep):
        pass

    def end_loop(self):
        pass

