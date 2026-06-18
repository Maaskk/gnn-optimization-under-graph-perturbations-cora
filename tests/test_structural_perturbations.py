import torch

from gnn_robustness.structural_perturbations import add_fake_edges, remove_edges


def test_remove_edges_removes_expected_fraction_deterministically():
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 0],
        ]
    )

    first = remove_edges(edge_index, severity=0.25, seed=42)
    second = remove_edges(edge_index, severity=0.25, seed=42)

    assert first.shape == (2, 6)
    assert torch.equal(first, second)


def test_add_fake_edges_avoids_self_loops_and_existing_edges():
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
        ]
    )

    perturbed = add_fake_edges(edge_index, num_nodes=6, severity=0.5, seed=7)
    original_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    added_edges = list(zip(perturbed[0].tolist()[4:], perturbed[1].tolist()[4:]))

    assert perturbed.shape == (2, 6)
    assert all(source != target for source, target in added_edges)
    assert all(edge not in original_edges for edge in added_edges)
    assert len(set(added_edges)) == len(added_edges)


def test_structural_perturbations_reject_invalid_severity():
    edge_index = torch.tensor([[0, 1], [1, 0]])

    functions = (
        remove_edges,
        lambda edges, severity, seed: add_fake_edges(edges, 2, severity, seed),
    )
    for fn in functions:
        try:
            fn(edge_index, severity=1.1, seed=1)
        except ValueError as exc:
            assert "severity" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid severity")

