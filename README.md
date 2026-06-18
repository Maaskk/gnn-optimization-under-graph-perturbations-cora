# Optimization of Graph Neural Networks under Graph Perturbations

**Project 13 - Option 4: Robustesse des GNN face au bruit dans le graphe**

This repository presents an experimental study of optimizer robustness for Graph Neural Networks on the Cora citation network. A two-layer Graph Convolutional Network is trained under a fixed protocol and evaluated with five optimizers:

- Adam
- AdamW
- RMSProp
- AdaGrad
- SGD

The study compares accuracy, macro F1-score, loss convergence, training time, and robustness under graph perturbations.

## Dataset

The experiments use the Cora Citation Network from the PyTorch Geometric Planetoid dataset.

| Property | Value |
| --- | ---: |
| Nodes | 2,708 |
| Directed citation edges | 10,556 |
| Node features | 1,433 |
| Classes | 7 |
| Training nodes | 140 |
| Validation nodes | 500 |
| Test nodes | 1,000 |

## Experimental Protocol

All optimizer comparisons use the same model and hyperparameters:

| Component | Setting |
| --- | --- |
| Model | 2-layer GCN |
| Hidden channels | 16 |
| Dropout | 0.5 |
| Learning rate | 0.01 |
| Weight decay | 0.0005 |
| Epochs | 200 |
| Seed | 42 |

Perturbation settings:

- Clean graph baseline
- Gaussian feature noise at severities 0.05, 0.10, 0.20, 0.30
- Random edge removal at severities 0.05, 0.10, 0.20, 0.30
- Fake edge addition at severities 0.05, 0.10, 0.20, 0.30

## Main Result

Adam is the strongest optimizer in the final aggregate comparison, with **0.702 mean test accuracy** and **0.694 mean macro F1** across all experiment rows. AdamW and RMSProp are also competitive, while SGD is much less stable under the fixed protocol.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `src/gnn_robustness/` | Dataset loading, GCN model, perturbations, training loop, metrics, plotting, and result aggregation |
| `scripts/` | Reproducible experiment and frontend data-generation scripts |
| `results/` | Final CSV result tables and exported figures |
| `reports/` | Final written report and supporting analysis |
| `docs/` | Interactive static frontend for visual exploration |
| `tests/` | Regression checks for metrics, perturbations, optimizers, result schema, and final outputs |

## Key Artifacts

- `reports/final_report.md`
- `results/all_results.csv`
- `results/final_optimizer_summary.csv`
- `results/clean_optimizer_results.csv`
- `results/feature_noise_results.csv`
- `results/edge_removal_results.csv`
- `results/fake_edge_addition_results.csv`
- `docs/index.html`

## Interactive Frontend

The static frontend in `docs/` turns the experiment outputs into an interactive visual report. It includes an animated Cora citation-network scene, optimizer ranking views, perturbation robustness curves, and loss convergence analysis.
