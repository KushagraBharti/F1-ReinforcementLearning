from pathlib import Path

from f1rl.imitation import (
    BehaviorCloningPolicy,
    collect_scripted_dataset,
    evaluate_policy,
    parse_args,
    run_imitation,
    train_behavior_cloning,
)


def test_behavior_cloning_smoke() -> None:
    observations, actions, summaries, dataset_meta = collect_scripted_dataset(episodes=1, max_steps=40, seed=7)
    assert observations.shape[0] == actions.shape[0]
    assert observations.shape[0] > 0
    assert summaries
    assert dataset_meta["samples"] == observations.shape[0]

    import torch

    device = torch.device("cpu")
    model, metrics = train_behavior_cloning(
        observations=observations,
        actions=actions,
        hidden_size=32,
        epochs=1,
        batch_size=16,
        learning_rate=1e-3,
        device=device,
        seed=7,
    )
    assert metrics["final_loss"] >= 0.0

    policy = BehaviorCloningPolicy(model=model.eval(), device=device)
    eval_summaries, aggregate = evaluate_policy(policy=policy, max_steps=40, seeds=[11])
    assert eval_summaries
    assert aggregate["episodes"] == 1


def test_run_imitation_exports_ppo_pretrain_module() -> None:
    args = parse_args(
        [
            "--episodes",
            "1",
            "--max-steps",
            "40",
            "--epochs",
            "1",
            "--batch-size",
            "16",
            "--device",
            "cpu",
            "--run-tag",
            "pytest",
        ]
    )
    summary_path, summary = run_imitation(args)
    assert summary_path.exists()
    assert Path(summary["model_path"]).exists()
    assert Path(summary["ppo_pretrain_module_path"]).exists()
    assert summary["ppo_pretrain"]["eval"]["aggregate"]["episodes"] == len(args.eval_seeds)
