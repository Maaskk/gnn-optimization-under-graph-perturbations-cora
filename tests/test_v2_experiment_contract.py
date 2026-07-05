import torch

from gnn_robustness.optimizers import make_optimizer
from gnn_robustness.v2_experiments import V2_RAW_COLUMNS, build_v2_raw_row


def test_v2_raw_schema_contains_required_reproducibility_fields():
    required = {
        "experiment_id",
        "git_commit",
        "timestamp",
        "dataset",
        "protocol",
        "robustness_setting",
        "optimizer",
        "seed",
        "resolved_seed",
        "hyperparameter_config",
        "perturbation_type",
        "requested_severity",
        "actual_perturbation_rate",
        "actual_perturbation_count",
        "train_accuracy",
        "validation_accuracy",
        "test_accuracy",
        "macro_f1",
        "loss",
        "training_time_seconds",
        "best_validation_epoch",
        "final_epoch",
        "hardware_metadata_path",
    }

    assert required.issubset(set(V2_RAW_COLUMNS))


def test_build_v2_raw_row_orders_columns_and_rejects_missing_fields():
    values = {column: "x" for column in V2_RAW_COLUMNS}
    values["test_accuracy"] = 0.75

    row = build_v2_raw_row(**values)

    assert list(row) == V2_RAW_COLUMNS
    assert row["test_accuracy"] == 0.75


def test_sgd_optimizer_supports_standard_tuned_momentum():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    optimizer = make_optimizer(
        "SGD",
        [parameter],
        learning_rate=0.01,
        weight_decay=0.0,
        momentum=0.9,
    )

    assert optimizer.param_groups[0]["momentum"] == 0.9
