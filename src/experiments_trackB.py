"""
Track B: Structural Graph Perturbation Experiments
- Edge removal robustness
- Fake edge addition robustness
Optimizers: Adam, AdamW, RMSProp, AdaGrad, SGD
"""

import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(__file__))

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from model import GCN
from perturbations import remove_edges, add_fake_edges
from train import get_optimizer, train_and_evaluate

# ── Hyperparameters ──────────────────────────────────────────────────────────
OPTIMIZERS      = ["Adam", "AdamW", "RMSProp", "AdaGrad", "SGD"]
SEVERITIES      = [0.05, 0.10, 0.20, 0.30]
SEEDS           = [42]
EPOCHS          = 200
HIDDEN_CHANNELS = 64
DROPOUT         = 0.5
LR              = 0.01
WEIGHT_DECAY    = 5e-4
TRACK           = "B"

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def load_cora():
    dataset = Planetoid(root="/tmp/Cora", name="Cora", transform=NormalizeFeatures())
    data = dataset[0]
    print(f"Cora loaded — Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}, "
          f"Features: {data.num_node_features}, Classes: {dataset.num_classes}")
    return data, dataset.num_node_features, dataset.num_classes


def run_experiment(data, num_features, num_classes, edge_index, perturbation_type, severity, seed):
    rows = []
    torch.manual_seed(seed)

    for opt_name in OPTIMIZERS:
        model = GCN(num_features, HIDDEN_CHANNELS, num_classes, DROPOUT)
        optimizer = get_optimizer(opt_name, model.parameters(), LR, WEIGHT_DECAY)

        acc, f1, loss, t, _ = train_and_evaluate(
            model, data, edge_index, optimizer, EPOCHS, DEVICE
        )

        row = {
            "track": TRACK,
            "optimizer": opt_name,
            "perturbation_type": perturbation_type,
            "severity": severity,
            "seed": seed,
            "epochs": EPOCHS,
            "hidden_channels": HIDDEN_CHANNELS,
            "dropout": DROPOUT,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "test_accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "final_loss": round(loss, 4),
            "training_time_seconds": round(t, 4),
        }
        rows.append(row)
        print(f"  [{perturbation_type} | sev={severity} | seed={seed}] "
              f"{opt_name:8s} → acc={acc:.4f}, f1={f1:.4f}, loss={loss:.4f}, t={t:.1f}s")

    return rows


def plot_metric_vs_severity(df, perturbation_type, metric, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    for opt in OPTIMIZERS:
        sub = df[df["optimizer"] == opt].sort_values("severity")
        ax.plot(sub["severity"], sub[metric], marker="o", label=opt)
    ax.set_xlabel("Severity")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(SEVERITIES)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved figure: {path}")


def main():
    data, num_features, num_classes = load_cora()

    # ── Track B-1: Edge Removal ───────────────────────────────────────────────
    print("\n=== Edge Removal Experiments ===")
    edge_removal_rows = []

    for severity in SEVERITIES:
        for seed in SEEDS:
            perturbed_ei = remove_edges(data.edge_index, severity, seed)
            print(f"  edge_removal sev={severity}: {data.edge_index.size(1)} → {perturbed_ei.size(1)} edges")
            rows = run_experiment(data, num_features, num_classes,
                                  perturbed_ei, "edge_removal", severity, seed)
            edge_removal_rows.extend(rows)

    df_er = pd.DataFrame(edge_removal_rows)
    er_path = os.path.join(RESULTS_DIR, "teammate_edge_removal_results.csv")
    df_er.to_csv(er_path, index=False)
    print(f"\nSaved: {er_path}")

    # ── Track B-2: Fake Edge Addition ─────────────────────────────────────────
    print("\n=== Fake Edge Addition Experiments ===")
    fake_edge_rows = []

    for severity in SEVERITIES:
        for seed in SEEDS:
            perturbed_ei = add_fake_edges(data.edge_index, data.num_nodes, severity, seed)
            print(f"  fake_edge sev={severity}: {data.edge_index.size(1)} → {perturbed_ei.size(1)} edges")
            rows = run_experiment(data, num_features, num_classes,
                                  perturbed_ei, "fake_edge_addition", severity, seed)
            fake_edge_rows.extend(rows)

    df_fe = pd.DataFrame(fake_edge_rows)
    fe_path = os.path.join(RESULTS_DIR, "teammate_fake_edge_results.csv")
    df_fe.to_csv(fe_path, index=False)
    print(f"\nSaved: {fe_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n=== Generating Figures ===")

    plot_metric_vs_severity(
        df_er, "edge_removal", "test_accuracy", "Test Accuracy",
        "Edge Removal — Accuracy vs Severity",
        "edge_removal_accuracy.png"
    )
    plot_metric_vs_severity(
        df_er, "edge_removal", "macro_f1", "Macro F1-Score",
        "Edge Removal — Macro F1 vs Severity",
        "edge_removal_f1.png"
    )
    plot_metric_vs_severity(
        df_fe, "fake_edge_addition", "test_accuracy", "Test Accuracy",
        "Fake Edge Addition — Accuracy vs Severity",
        "fake_edge_accuracy.png"
    )
    plot_metric_vs_severity(
        df_fe, "fake_edge_addition", "macro_f1", "Macro F1-Score",
        "Fake Edge Addition — Macro F1 vs Severity",
        "fake_edge_f1.png"
    )

    print("\n✅ Track B complete.")


if __name__ == "__main__":
    main()