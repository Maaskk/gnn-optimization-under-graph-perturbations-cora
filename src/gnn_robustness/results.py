RESULT_COLUMNS = [
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


def build_result_row(**values) -> dict:
    """Return a row ordered according to the shared project CSV contract."""

    missing = [column for column in RESULT_COLUMNS if column not in values]
    if missing:
        raise ValueError(f"Missing result columns: {missing}")
    return {column: values[column] for column in RESULT_COLUMNS}

