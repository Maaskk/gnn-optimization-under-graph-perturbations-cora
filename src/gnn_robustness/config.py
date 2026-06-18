from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Hyperparameters shared across optimizer comparisons."""

    epochs: int = 200
    hidden_channels: int = 16
    dropout: float = 0.5
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    seeds: tuple[int, ...] = (42,)
    severities: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    device: str = "cpu"

