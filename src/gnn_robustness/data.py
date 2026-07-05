from pathlib import Path

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def load_planetoid(name: str, root: str | Path = "data"):
    """Load a supported Planetoid citation network."""

    if name not in {"Cora", "CiteSeer", "PubMed"}:
        raise ValueError("Supported Planetoid datasets are Cora, CiteSeer, and PubMed")
    dataset = Planetoid(root=str(root), name=name, transform=NormalizeFeatures())
    return dataset, dataset[0]


def load_cora(root: str | Path = "data"):
    """Load the professor-requested Cora citation network."""

    return load_planetoid("Cora", root=root)


def cora_summary(dataset, data) -> dict[str, int]:
    """Return stable dataset facts useful for reports."""

    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_features": int(dataset.num_node_features),
        "num_classes": int(dataset.num_classes),
        "train_nodes": int(data.train_mask.sum().item()),
        "validation_nodes": int(data.val_mask.sum().item()),
        "test_nodes": int(data.test_mask.sum().item()),
    }
