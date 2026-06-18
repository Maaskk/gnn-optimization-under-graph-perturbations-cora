import torch


def remove_edges(edge_index, severity: float, seed: int = 42) -> torch.Tensor:
    """
    Randomly remove a fraction `severity` of existing edges.
    Works on undirected graphs stored as symmetric edge_index.
    """
    torch.manual_seed(seed)
    num_edges = edge_index.size(1)
    num_to_remove = int(num_edges * severity)
    perm = torch.randperm(num_edges)
    keep_mask = torch.ones(num_edges, dtype=torch.bool)
    keep_mask[perm[:num_to_remove]] = False
    return edge_index[:, keep_mask]


def add_fake_edges(edge_index, num_nodes: int, severity: float, seed: int = 42) -> torch.Tensor:
    """
    Add random fake edges equal to `severity` * existing_edges.
    Avoids duplicate edges (best-effort).
    """
    torch.manual_seed(seed)
    num_existing = edge_index.size(1)
    num_to_add = int(num_existing * severity)

    # Build a set of existing edges for fast lookup
    existing = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    src_list, dst_list = [], []
    attempts = 0
    max_attempts = num_to_add * 10

    while len(src_list) < num_to_add and attempts < max_attempts:
        s = torch.randint(0, num_nodes, (1,)).item()
        d = torch.randint(0, num_nodes, (1,)).item()
        if s != d and (s, d) not in existing:
            src_list.append(s)
            dst_list.append(d)
            existing.add((s, d))
        attempts += 1

    if src_list:
        fake_edges = torch.tensor([src_list, dst_list], dtype=edge_index.dtype)
        return torch.cat([edge_index, fake_edges], dim=1)
    return edge_index