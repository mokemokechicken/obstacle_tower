from tower.const import Action


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
    _original_to_limited: dict = None

    @classmethod
    def from_int(cls, n):
        assert 0 <= n < cls.size
        return cls.actions[n]

    @classmethod
    def original_action_to_limited_action(cls, n):
        if cls._original_to_limited is None:
            cls._original_to_limited = cls._make_original_to_limited_map()
        return cls._original_to_limited.get(n, 0)

    @classmethod
    def _make_original_to_limited_map(cls):
        ret = {}
        for a in range(Action.size):
            orig_action = Action.from_int(a)
            for k, la in enumerate(cls.actions):
                if list(orig_action) == list(la):
                    ret[a] = k
                    break
        return ret
