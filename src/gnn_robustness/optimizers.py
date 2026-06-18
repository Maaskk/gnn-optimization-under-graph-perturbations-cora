from collections.abc import Iterable

import torch

DEFAULT_OPTIMIZERS = ["Adam", "AdamW", "RMSProp", "AdaGrad", "SGD"]
BONUS_OPTIMIZERS = ["AMSGrad"]


def make_optimizer(
    name: str,
    parameters: Iterable[torch.nn.Parameter],
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Create a PyTorch optimizer from the project optimizer list."""

    normalized = name.lower()
    if normalized == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if normalized == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if normalized == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay)
    if normalized == "adagrad":
        return torch.optim.Adagrad(parameters, lr=learning_rate, weight_decay=weight_decay)
    if normalized == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay)
    if normalized == "amsgrad":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            amsgrad=True,
        )
    supported = ", ".join(DEFAULT_OPTIMIZERS + BONUS_OPTIMIZERS)
    raise ValueError(f"Unsupported optimizer '{name}'. Supported optimizers: {supported}")

