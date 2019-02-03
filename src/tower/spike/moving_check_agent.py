from collections import namedtuple

from obstacle_tower_env import ObstacleTowerEnv

from tower.spike.const import Action


CheckResult = namedtuple('CheckResult', 'frame0 frame1 action')


class MovingCheckAgent:
    def __init__(self, env: ObstacleTowerEnv):
        self.env = env
        self.n_step = 20
        self.check_actions = [Action.NOP, Action.FORWARD, Action.BACK,
                              Action.CAMERA_RIGHT, Action.LEFT,
                              Action.RIGHT, Action.LEFT]
        self.state = CheckState()
        self.done = False
        self.results = []

    def step(self):
        if self.done:
            return

        action = self.check_actions[self.state.action_index]
        self.state.current_step += 1
        if self.state.current_step == self.n_step:
            self.state.current_step = 0
            self.state.action_index += 1
            if self.state.action_index >= len(self.check_actions):
                self.done = True

        frame0 = self.env.render()
        self.env.step(action)
        frame1 = self.env.render()
        self.results.append(CheckResult(frame0, frame1, action))


class CheckState:
    action_index: int = 0
    current_step: int = 0
