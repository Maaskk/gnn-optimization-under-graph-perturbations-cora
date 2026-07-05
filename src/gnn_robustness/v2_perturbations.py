from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FeatureMaskingResult:
    features: torch.Tensor
    metadata: dict[str, float | int]


@dataclass(frozen=True)
class EdgePerturbationResult:
    edge_index: torch.Tensor
    metadata: dict[str, float | int]


def _validate_rate(severity: float) -> None:
    if severity < 0 or severity > 1:
        raise ValueError("severity must be between 0 and 1")


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def unique_undirected_edges(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    """Return unique non-self-loop undirected edges as sorted node-id pairs."""

    pairs: set[tuple[int, int]] = set()
    for source, target in zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=False):
        source = int(source)
        target = int(target)
        if source == target:
            continue
        pairs.add((source, target) if source < target else (target, source))
    return pairs


def _edge_index_from_pairs(
    pairs: set[tuple[int, int]],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    directed_edges: list[tuple[int, int]] = []
    for source, target in sorted(pairs):
        directed_edges.append((source, target))
        directed_edges.append((target, source))
    if not directed_edges:
        return torch.empty((2, 0), dtype=dtype, device=device)
    return torch.tensor(directed_edges, dtype=dtype, device=device).t().contiguous()


def mask_active_features(
    features: torch.Tensor, severity: float, seed: int
) -> FeatureMaskingResult:
    """Mask a fraction of currently active/non-zero feature entries to zero."""

    _validate_rate(severity)
    masked = features.clone()
    active_positions = torch.nonzero(features != 0, as_tuple=False).cpu()
    active_count = int(active_positions.size(0))
    requested_count = int(active_count * severity)
    if requested_count > 0:
        permutation = torch.randperm(active_count, generator=_generator(seed))
        chosen = active_positions[permutation[:requested_count]].to(features.device)
        masked[chosen[:, 0], chosen[:, 1]] = 0
    actual_rate = 0.0 if active_count == 0 else requested_count / active_count
    return FeatureMaskingResult(
        features=masked,
        metadata={
            "requested_masking_rate": float(severity),
            "active_feature_entries": active_count,
            "masked_feature_entries": requested_count,
            "actual_masking_rate": float(actual_rate),
        },
    )


def remove_undirected_edges(
    edge_index: torch.Tensor,
    severity: float,
    seed: int,
) -> EdgePerturbationResult:
    """Remove a fraction of unique undirected connections and return symmetric edges."""

    _validate_rate(severity)
    pairs = unique_undirected_edges(edge_index)
    original_count = len(pairs)
    requested = int(original_count * severity)
    pair_list = sorted(pairs)
    if requested > 0:
        permutation = torch.randperm(original_count, generator=_generator(seed)).tolist()
        to_remove = {pair_list[index] for index in permutation[:requested]}
    else:
        to_remove = set()
    kept = pairs - to_remove
    actual_rate = 0.0 if original_count == 0 else len(to_remove) / original_count
    return EdgePerturbationResult(
        edge_index=_edge_index_from_pairs(kept, dtype=edge_index.dtype, device=edge_index.device),
        metadata={
            "original_unique_edges": original_count,
            "requested_removed_edges": requested,
            "actual_removed_edges": len(to_remove),
            "actual_removed_rate": float(actual_rate),
        },
    )


def add_undirected_fake_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    severity: float,
    seed: int,
) -> EdgePerturbationResult:
    """Add random non-overlapping undirected fake edges, stored symmetrically."""

    _validate_rate(severity)
    original_pairs = unique_undirected_edges(edge_index)
    original_count = len(original_pairs)
    requested = int(original_count * severity)
    all_pairs = {
        (source, target)
        for source in range(num_nodes)
        for target in range(source + 1, num_nodes)
        if (source, target) not in original_pairs
    }
    if requested > len(all_pairs):
        raise RuntimeError(
            f"Requested {requested} fake edges but only {len(all_pairs)} are available"
        )
    candidates = sorted(all_pairs)
    if requested > 0:
        permutation = torch.randperm(len(candidates), generator=_generator(seed)).tolist()
        inserted = {candidates[index] for index in permutation[:requested]}
    else:
        inserted = set()
    combined = original_pairs | inserted
    actual_rate = 0.0 if original_count == 0 else len(inserted) / original_count
    return EdgePerturbationResult(
        edge_index=_edge_index_from_pairs(
            combined, dtype=edge_index.dtype, device=edge_index.device
        ),
        metadata={
            "original_unique_edges": original_count,
            "requested_inserted_edges": requested,
            "actual_inserted_edges": len(inserted),
            "actual_inserted_rate": float(actual_rate),
        },
    )
