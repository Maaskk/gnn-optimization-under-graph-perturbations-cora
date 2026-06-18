import torch


def _validate_severity(severity: float) -> None:
    if severity < 0 or severity > 1:
        raise ValueError("severity must be between 0 and 1")


def remove_edges(edge_index: torch.Tensor, severity: float, seed: int) -> torch.Tensor:
    """Randomly remove a severity fraction of directed edges."""

    _validate_severity(severity)
    if severity == 0:
        return edge_index.clone()

    num_edges = edge_index.size(1)
    num_to_remove = int(num_edges * severity)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(num_edges, generator=generator)
    keep_mask = torch.ones(num_edges, dtype=torch.bool)
    keep_mask[permutation[:num_to_remove]] = False
    return edge_index[:, keep_mask]


def add_fake_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    severity: float,
    seed: int,
) -> torch.Tensor:
    """Add random non-existing directed edges without self-loops."""

    _validate_severity(severity)
    if severity == 0:
        return edge_index.clone()

    num_existing = edge_index.size(1)
    num_to_add = int(num_existing * severity)
    existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    added_edges: list[tuple[int, int]] = []

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    max_attempts = max(100, num_to_add * 100)
    attempts = 0

    while len(added_edges) < num_to_add and attempts < max_attempts:
        source = int(torch.randint(0, num_nodes, (1,), generator=generator).item())
        target = int(torch.randint(0, num_nodes, (1,), generator=generator).item())
        candidate = (source, target)
        if source != target and candidate not in existing_edges:
            existing_edges.add(candidate)
            added_edges.append(candidate)
        attempts += 1

    if len(added_edges) < num_to_add:
        raise RuntimeError(
            f"Could only add {len(added_edges)} fake edges out of requested {num_to_add}"
        )

    fake_edges = torch.tensor(added_edges, dtype=edge_index.dtype).t().contiguous()
    return torch.cat([edge_index, fake_edges], dim=1)

