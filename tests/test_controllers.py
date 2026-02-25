import numpy as np

from f1rl.controllers import ScriptedController


def test_scripted_controller_is_deterministic() -> None:
    controller_a = ScriptedController()
    controller_b = ScriptedController()

    observation = np.array([-0.2, 0.0, 0.2, 0.4, 0.2, 0.0, -0.2, -0.1], dtype=np.float32)
    actions_a = [controller_a.action(observation) for _ in range(5)]
    actions_b = [controller_b.action(observation) for _ in range(5)]

    for left, right in zip(actions_a, actions_b):
        assert np.allclose(left, right)


def test_scripted_controller_reset_restores_sequence() -> None:
    controller = ScriptedController()
    observation = np.array([0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, 0.0], dtype=np.float32)

    first = controller.action(observation)
    second = controller.action(observation)
    controller.reset()
    first_after_reset = controller.action(observation)
    second_after_reset = controller.action(observation)

    assert np.allclose(first, first_after_reset)
    assert np.allclose(second, second_after_reset)
