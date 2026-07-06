import torch

from gnn_robustness.v2_perturbations import (
    add_undirected_fake_edges,
    mask_active_features,
    remove_undirected_edges,
    sample_absent_undirected_edges,
    unique_undirected_edges,
)


def test_feature_masking_masks_requested_fraction_of_active_entries_only():
    features = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [3.0, 0.0, 4.0],
            [0.0, 0.0, 5.0],
        ]
    )

    result = mask_active_features(features, severity=0.4, seed=123)

    active_before = features != 0
    changed = result.features != features
    assert result.metadata["requested_masking_rate"] == 0.4
    assert result.metadata["active_feature_entries"] == 5
    assert result.metadata["masked_feature_entries"] == 2
    assert result.metadata["actual_masking_rate"] == 0.4
    assert torch.all(changed <= active_before)
    assert int((result.features == 0).sum()) == int((features == 0).sum()) + 2


def test_feature_masking_is_reproducible_for_same_seed():
    features = torch.eye(8)

    first = mask_active_features(features, severity=0.25, seed=77).features
    second = mask_active_features(features, severity=0.25, seed=77).features
    third = mask_active_features(features, severity=0.25, seed=78).features

    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_remove_undirected_edges_preserves_symmetric_representation():
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ]
    )

    result = remove_undirected_edges(edge_index, severity=0.5, seed=5)
    pairs = unique_undirected_edges(result.edge_index)

    assert result.metadata["original_unique_edges"] == 4
    assert result.metadata["requested_removed_edges"] == 2
    assert result.metadata["actual_removed_edges"] == 2
    assert result.metadata["actual_removed_rate"] == 0.5
    assert len(pairs) == 2
    assert result.edge_index.size(1) == 4
    assert unique_undirected_edges(result.edge_index) == {
        tuple(sorted(edge))
        for edge in zip(result.edge_index[0].tolist(), result.edge_index[1].tolist(), strict=False)
    }


def test_fake_edge_addition_avoids_self_loops_duplicates_and_original_edges():
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ]
    )
    original_pairs = unique_undirected_edges(edge_index)

    result = add_undirected_fake_edges(edge_index, num_nodes=5, severity=1.0, seed=11)
    new_pairs = unique_undirected_edges(result.edge_index)
    added_pairs = new_pairs - original_pairs

    assert result.metadata["original_unique_edges"] == 2
    assert result.metadata["requested_inserted_edges"] == 2
    assert result.metadata["actual_inserted_edges"] == 2
    assert result.metadata["actual_inserted_rate"] == 1.0
    assert len(added_pairs) == 2
    assert all(source != target for source, target in added_pairs)
    assert added_pairs.isdisjoint(original_pairs)
    assert result.edge_index.size(1) == 8


def test_absent_edge_sampler_supports_sparse_large_graphs_without_candidate_enumeration():
    original_pairs = {(0, 1), (2, 3), (7, 11)}

    sampled = sample_absent_undirected_edges(
        original_pairs=original_pairs,
        num_nodes=20_000,
        requested=8,
        seed=1234,
    )

    assert len(sampled) == 8
    assert sampled.isdisjoint(original_pairs)
    assert all(source < target for source, target in sampled)
    assert all(source != target for source, target in sampled)
