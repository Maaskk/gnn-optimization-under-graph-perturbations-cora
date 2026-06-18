import torch

from gnn_robustness.optimizers import DEFAULT_OPTIMIZERS, make_optimizer


def test_default_optimizers_match_project_title():
    assert DEFAULT_OPTIMIZERS == ["Adam", "AdamW", "RMSProp", "AdaGrad", "SGD"]


def test_make_optimizer_configures_sgd_learning_rate_and_weight_decay():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    optimizer = make_optimizer(
        "SGD",
        [parameter],
        learning_rate=0.05,
        weight_decay=0.01,
    )

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["lr"] == 0.05
    assert optimizer.defaults["weight_decay"] == 0.01


def test_make_optimizer_supports_amsgrad_as_bonus_optimizer():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    optimizer = make_optimizer(
        "AMSGrad",
        [parameter],
        learning_rate=0.01,
        weight_decay=0.001,
    )

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["amsgrad"] is True


def test_make_optimizer_rejects_unknown_name():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    try:
        make_optimizer("NotAnOptimizer", [parameter], learning_rate=0.01, weight_decay=0.0)
    except ValueError as exc:
        assert "Unsupported optimizer" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported optimizer")

