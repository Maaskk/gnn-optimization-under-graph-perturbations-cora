#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_robustness.data import load_planetoid  # noqa: E402
from gnn_robustness.model import GCN  # noqa: E402
from gnn_robustness.optimizers import make_optimizer  # noqa: E402
from gnn_robustness.train import set_seed  # noqa: E402
from gnn_robustness.v2_perturbations import mask_active_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a real hidden-embedding PCA artifact.")
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="results/v2/embedding/cora_adam_seed42_clean_pca.json")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset, data = load_planetoid(args.dataset, args.data_root)
    model = GCN(dataset.num_node_features, 16, dataset.num_classes, dropout=0.5)
    optimizer = make_optimizer(
        args.optimizer, model.parameters(), learning_rate=0.01, weight_decay=0.0005
    )
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        hidden = model.encode(data.x, data.edge_index).cpu().numpy()
        clean_logits = model(data.x, data.edge_index)
        masked_features = mask_active_features(data.x, severity=0.2, seed=args.seed).features
        masked_logits = model(masked_features, data.edge_index)

    centered = hidden - hidden.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    payload = {
        "artifact": "actual_hidden_gcn_embedding_pca",
        "dataset": args.dataset,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "epochs": args.epochs,
        "condition": "clean_trained_clean_hidden_pca",
        "perturbed_prediction_condition": "feature_masking_0.20",
        "placement_note": (
            "Les positions des noeuds sont les coordonnées PCA de représentations cachées "
            "réelles du GCN, pas les étiquettes vraies."
        ),
        "nodes": [
            {
                "id": int(index),
                "x": float(coords[index, 0]),
                "y": float(coords[index, 1]),
                "true_label": int(data.y[index]),
                "pred_clean": int(clean_logits[index].argmax().item()),
                "pred_feature_masking_20": int(masked_logits[index].argmax().item()),
                "split": "train"
                if bool(data.train_mask[index])
                else "validation"
                if bool(data.val_mask[index])
                else "test"
                if bool(data.test_mask[index])
                else "unlabeled",
            }
            for index in range(int(data.num_nodes))
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote embedding artifact: {output}")


if __name__ == "__main__":
    main()
