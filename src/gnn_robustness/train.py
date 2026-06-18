import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .metrics import accuracy_score, macro_f1_score
from .model import GCN
from .optimizers import make_optimizer
from .perturbations import add_feature_noise
from .results import build_result_row


@dataclass(frozen=True)
class TrainingOutcome:
    result_row: dict
    history_rows: list[dict]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_metrics(model: GCN, data, mask: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[mask], data.y[mask])
    return {
        "loss": float(loss.item()),
        "accuracy": accuracy_score(logits, data.y, mask),
        "macro_f1": macro_f1_score(logits, data.y, mask),
    }


def train_one_epoch(model: GCN, data, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


def run_training(
    data,
    num_features: int,
    num_classes: int,
    optimizer_name: str,
    config: ExperimentConfig,
    perturbation_type: str,
    severity: float,
    seed: int,
) -> TrainingOutcome:
    """Train one GCN run and return result plus convergence history."""

    set_seed(seed)
    run_data = data.clone().to(config.device)
    if perturbation_type == "feature_noise":
        run_data.x = add_feature_noise(run_data.x, severity=severity, seed=seed)
    elif perturbation_type != "clean":
        raise ValueError(f"Unsupported perturbation_type for Ossama track: {perturbation_type}")

    model = GCN(
        input_channels=num_features,
        hidden_channels=config.hidden_channels,
        output_channels=num_classes,
        dropout=config.dropout,
    ).to(config.device)
    optimizer = make_optimizer(
        optimizer_name,
        model.parameters(),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    start = time.perf_counter()
    history_rows: list[dict] = []
    final_train_loss = 0.0
    for epoch in range(1, config.epochs + 1):
        final_train_loss = train_one_epoch(model, run_data, optimizer)
        val_metrics = _split_metrics(model, run_data, run_data.val_mask)
        history_rows.append(
            {
                "track": "ossama",
                "optimizer": optimizer_name,
                "perturbation_type": perturbation_type,
                "severity": severity,
                "seed": seed,
                "epoch": epoch,
                "train_loss": final_train_loss,
                "validation_loss": val_metrics["loss"],
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": val_metrics["macro_f1"],
            }
        )
    training_time = time.perf_counter() - start
    test_metrics = _split_metrics(model, run_data, run_data.test_mask)

    result_row = build_result_row(
        track="ossama",
        optimizer=optimizer_name,
        perturbation_type=perturbation_type,
        severity=severity,
        seed=seed,
        epochs=config.epochs,
        hidden_channels=config.hidden_channels,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        test_accuracy=test_metrics["accuracy"],
        macro_f1=test_metrics["macro_f1"],
        final_loss=final_train_loss,
        training_time_seconds=training_time,
    )
    return TrainingOutcome(result_row=result_row, history_rows=history_rows)

