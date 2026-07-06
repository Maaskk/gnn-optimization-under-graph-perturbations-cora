from gnn_robustness.v2_config import load_v2_config
from gnn_robustness.v2_tuning import (
    apply_locked_hyperparameters,
    load_locked_hyperparameters,
    optimizer_tuning_grid,
)


def test_optimizer_tuning_grid_includes_sgd_momentum_values_only_for_sgd():
    config = load_v2_config("configs/v2_tuned_cora.yaml")

    adam_grid = optimizer_tuning_grid(config, "Adam")
    sgd_grid = optimizer_tuning_grid(config, "SGD")

    assert {trial["momentum"] for trial in adam_grid} == {0.0}
    assert {trial["momentum"] for trial in sgd_grid} == {0.0, 0.9}
    assert len(sgd_grid) == len(adam_grid) * 2


def test_apply_locked_hyperparameters_updates_training_without_using_test_scores():
    config = load_v2_config("configs/v2_tuned_cora.yaml")
    locked = {
        "Adam": {
            "learning_rate": 0.003,
            "weight_decay": 0.0,
            "momentum": 0.0,
            "selection_metric": "mean_validation_accuracy",
            "uses_test_split": False,
        },
        "SGD": {
            "learning_rate": 0.03,
            "weight_decay": 0.005,
            "momentum": 0.9,
            "selection_metric": "mean_validation_accuracy",
            "uses_test_split": False,
        },
    }

    adam_config, adam_momentum = apply_locked_hyperparameters(config, "Adam", locked)
    sgd_config, sgd_momentum = apply_locked_hyperparameters(config, "SGD", locked)

    assert adam_config.training.learning_rate == 0.003
    assert adam_config.training.weight_decay == 0.0
    assert adam_momentum == 0.0
    assert sgd_config.training.learning_rate == 0.03
    assert sgd_config.training.weight_decay == 0.005
    assert sgd_momentum == 0.9


def test_locked_hyperparameters_reject_test_split_selection(tmp_path):
    config = load_v2_config("configs/v2_tuned_cora.yaml")
    path = tmp_path / "locked.json"
    path.write_text(
        '{"Adam": {"learning_rate": 0.01, "weight_decay": 0.0, "uses_test_split": true}}',
        encoding="utf-8",
    )

    locked = load_locked_hyperparameters(path)

    try:
        apply_locked_hyperparameters(config, "Adam", locked)
    except ValueError as exc:
        assert "test split" in str(exc)
    else:
        raise AssertionError("Expected test-split locked settings to be rejected")
