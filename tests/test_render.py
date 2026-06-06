from f1rl.config import action_to_controls
from f1rl.render import HUD_MARGIN_PX, PygameRenderer, _hud_origin


class _KeyState(dict[int, bool]):
    def __getitem__(self, key: int) -> bool:
        return self.get(key, False)


class _FakeKey:
    def __init__(self, pressed: _KeyState) -> None:
        self._pressed = pressed

    def get_pressed(self) -> _KeyState:
        return self._pressed


class _FakePygame:
    K_w = 1
    K_UP = 2
    K_s = 3
    K_DOWN = 4
    K_a = 5
    K_LEFT = 6
    K_d = 7
    K_RIGHT = 8

    def __init__(self, pressed: _KeyState) -> None:
        self.key = _FakeKey(pressed)


def _keyboard_action(pressed: dict[int, bool]) -> int:
    renderer = PygameRenderer.__new__(PygameRenderer)
    renderer.pygame = _FakePygame(_KeyState(pressed))
    return renderer.keyboard_action()


def test_manual_keyboard_left_right_are_swapped_for_screen_controls() -> None:
    left_action = _keyboard_action({_FakePygame.K_LEFT: True})
    right_action = _keyboard_action({_FakePygame.K_RIGHT: True})
    left_steer = action_to_controls(left_action)[2]
    right_steer = action_to_controls(right_action)[2]

    assert left_action == 4
    assert right_action == 3
    assert left_steer > 0.0
    assert right_steer < 0.0


def test_manual_keyboard_combined_left_right_controls_are_swapped() -> None:
    throttle_left = _keyboard_action({_FakePygame.K_UP: True, _FakePygame.K_LEFT: True})
    throttle_right = _keyboard_action({_FakePygame.K_UP: True, _FakePygame.K_RIGHT: True})
    brake_left = _keyboard_action({_FakePygame.K_DOWN: True, _FakePygame.K_LEFT: True})
    brake_right = _keyboard_action({_FakePygame.K_DOWN: True, _FakePygame.K_RIGHT: True})

    assert throttle_left == 6
    assert throttle_right == 5
    assert brake_left == 8
    assert brake_right == 7


def test_hud_origin_uses_free_right_side_when_available() -> None:
    x, y = _hud_origin((1470, 775), [(360, 16), (260, 16)])

    assert x == 1470 - 360 - HUD_MARGIN_PX
    assert y == HUD_MARGIN_PX


def test_hud_origin_falls_back_to_margin_for_small_windows() -> None:
    x, y = _hud_origin((320, 240), [(360, 16)])

    assert x == HUD_MARGIN_PX
    assert y == HUD_MARGIN_PX
