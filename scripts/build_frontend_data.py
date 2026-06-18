from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA_DIR = ROOT / "docs" / "assets" / "data"

sys.path.insert(0, str(SRC))

from gnn_robustness.data import load_cora  # noqa: E402


CSV_FILES = [
    "all_results.csv",
    "final_optimizer_summary.csv",
    "clean_loss_history.csv",
    "cora_dataset_summary.csv",
]


def copy_result_csvs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename in CSV_FILES:
        source = ROOT / "results" / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing required result file: {source}")
        shutil.copy2(source, DATA_DIR / filename)


def choose_sample_nodes(labels: np.ndarray, degree: np.ndarray, max_nodes: int = 230) -> list[int]:
    rng = random.Random(42)
    chosen: list[int] = []

    for node in np.argsort(-degree)[:85]:
        chosen.append(int(node))

    for class_id in sorted(set(labels.tolist())):
        class_nodes = [int(index) for index in np.where(labels == class_id)[0]]
        rng.shuffle(class_nodes)
        chosen.extend(class_nodes[:24])

    ordered_unique = []
    seen = set()
    for node in chosen:
        if node in seen:
            continue
        ordered_unique.append(node)
        seen.add(node)
        if len(ordered_unique) >= max_nodes:
            break
    return ordered_unique


def build_graph_sample() -> dict[str, object]:
    dataset, data = load_cora(ROOT / "data")
    edge_index = data.edge_index.cpu().numpy()
    labels = data.y.cpu().numpy()
    out_degree = np.bincount(edge_index[0], minlength=data.num_nodes)
    in_degree = np.bincount(edge_index[1], minlength=data.num_nodes)
    degree = out_degree + in_degree
    max_degree = max(1, int(degree.max()))
    selected_nodes = choose_sample_nodes(labels, degree)
    selected = set(selected_nodes)
    rng = random.Random(13)

    nodes = []
    class_count = int(dataset.num_classes)
    for node_id in selected_nodes:
        class_id = int(labels[node_id])
        angle = (2 * np.pi * class_id / class_count) - np.pi / 2
        radius = 0.23 + 0.17 * (1 - min(1, degree[node_id] / max_degree))
        jitter_x = rng.uniform(-0.09, 0.09)
        jitter_y = rng.uniform(-0.08, 0.08)
        x = 0.54 + np.cos(angle) * radius + jitter_x
        y = 0.5 + np.sin(angle) * radius * 0.8 + jitter_y
        split = "unlabeled"
        if bool(data.train_mask[node_id]):
            split = "train"
        elif bool(data.val_mask[node_id]):
            split = "validation"
        elif bool(data.test_mask[node_id]):
            split = "test"

        nodes.append(
            {
                "id": int(node_id),
                "original_id": int(node_id),
                "class_id": class_id,
                "degree": int(degree[node_id]),
                "degree_rank": round(float(degree[node_id] / max_degree), 4),
                "split": split,
                "x": round(float(np.clip(x, 0.08, 0.92)), 4),
                "y": round(float(np.clip(y, 0.1, 0.9)), 4),
                "phase": round(rng.random() * 6.283, 4),
            }
        )

    edges = []
    for source, target in zip(edge_index[0], edge_index[1], strict=False):
        source = int(source)
        target = int(target)
        if source in selected and target in selected:
            edges.append({"source": source, "target": target})
        if len(edges) >= 760:
            break

    return {
        "source": "PyTorch Geometric Planetoid Cora dataset sample",
        "num_nodes_total": int(data.num_nodes),
        "num_edges_total": int(data.edge_index.size(1)),
        "nodes": nodes,
        "edges": edges,
    }


def main() -> None:
    copy_result_csvs()
    graph_sample = build_graph_sample()
    with (DATA_DIR / "cora_graph_sample.json").open("w", encoding="utf-8") as handle:
        json.dump(graph_sample, handle, indent=2)
        handle.write("\n")
    print(f"Wrote frontend data bundle to {DATA_DIR}")


if __name__ == "__main__":
    main()
