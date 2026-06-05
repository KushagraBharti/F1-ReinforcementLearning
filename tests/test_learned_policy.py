import numpy as np
import torch

from f1rl.learned_policy import (
    PolicyNormalizer,
    SACActor,
    load_policy_checkpoint,
    save_policy_checkpoint,
)


def test_sac_actor_control_modes_are_explicit() -> None:
    raw = torch.tensor([[0.0, 0.5, -0.25]], dtype=torch.float32)
    independent = SACActor(4, hidden_sizes=(8, 8), control_mode="independent")
    dominance = SACActor(4, hidden_sizes=(8, 8), control_mode="dominance")

    independent_action = independent.raw_to_controls(raw)
    dominance_action = dominance.raw_to_controls(raw)

    assert torch.allclose(independent_action, torch.tensor([[0.5, 0.75, -0.25]]))
    assert torch.allclose(dominance_action, torch.tensor([[0.125, 0.75, -0.25]]))


def test_sac_actor_checkpoint_preserves_control_mode(tmp_path) -> None:
    actor = SACActor(4, hidden_sizes=(), control_mode="dominance")
    normalizer = PolicyNormalizer(mean=np.zeros(4, dtype=np.float32), std=np.ones(4, dtype=np.float32))
    path = tmp_path / "policy.pt"

    save_policy_checkpoint(path, actor=actor, normalizer=normalizer, metadata={"stage": "unit"})
    loaded_actor, loaded_normalizer, metadata = load_policy_checkpoint(path)

    assert loaded_actor.control_mode == "dominance"
    assert loaded_actor.hidden_sizes == ()
    assert metadata["stage"] == "unit"
    assert np.allclose(loaded_normalizer.mean, normalizer.mean)
