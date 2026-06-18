import math

import torch

from gnn_robustness.metrics import accuracy_score, macro_f1_score


def test_accuracy_score_uses_masked_nodes_only():
    logits = torch.tensor(
        [
            [3.0, 0.1],
            [0.2, 2.0],
            [4.0, 0.1],
            [0.1, 5.0],
        ]
    )
    labels = torch.tensor([0, 1, 1, 0])
    mask = torch.tensor([True, True, True, False])

    assert accuracy_score(logits, labels, mask) == 2 / 3


def test_macro_f1_score_uses_masked_nodes_only():
    logits = torch.tensor(
        [
            [3.0, 0.1],
            [2.0, 0.2],
            [0.1, 4.0],
            [5.0, 0.1],
        ]
    )
    labels = torch.tensor([0, 1, 1, 0])
    mask = torch.tensor([True, True, True, False])

    assert math.isclose(macro_f1_score(logits, labels, mask), 2 / 3, rel_tol=1e-9)

