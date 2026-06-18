from gnn_robustness.results import RESULT_COLUMNS, build_result_row


def test_result_columns_match_shared_contract():
    assert RESULT_COLUMNS == [
        "track",
        "optimizer",
        "perturbation_type",
        "severity",
        "seed",
        "epochs",
        "hidden_channels",
        "dropout",
        "learning_rate",
        "weight_decay",
        "test_accuracy",
        "macro_f1",
        "final_loss",
        "training_time_seconds",
    ]


def test_build_result_row_orders_values_for_csv_writer():
    row = build_result_row(
        track="baseline",
        optimizer="Adam",
        perturbation_type="clean",
        severity=0.0,
        seed=42,
        epochs=200,
        hidden_channels=16,
        dropout=0.5,
        learning_rate=0.01,
        weight_decay=0.0005,
        test_accuracy=0.81,
        macro_f1=0.80,
        final_loss=0.42,
        training_time_seconds=12.5,
    )

    assert list(row.keys()) == RESULT_COLUMNS
    assert row["optimizer"] == "Adam"
    assert row["perturbation_type"] == "clean"
    assert row["test_accuracy"] == 0.81
