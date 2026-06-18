# Optimization of Graph Neural Networks under Graph Perturbations

Project 13 - Option 4: Robustesse des GNN face au bruit dans le graphe

## Full Title

Optimization of Graph Neural Networks under Graph Perturbations: Adam vs AdamW vs RMSProp vs AdaGrad vs SGD on the Cora Citation Network

## Objective

This project studies how optimization algorithms affect the training stability and robustness of Graph Neural Networks under graph perturbations.

We train a Graph Convolutional Network (GCN) on the Cora citation network and compare:

- Adam
- AdamW
- RMSProp
- AdaGrad
- SGD

The comparison uses accuracy, macro F1-score, loss convergence, training time, and robustness under perturbations.

## Dataset

Cora Citation Network

## Perturbations

- Edge removal
- Fake edge addition
- Feature noise

## Suggested Repository Structure

```text
.
├── data/                 # Optional cached dataset files
├── notebooks/            # Exploratory notebooks and plots
├── results/              # Metrics, figures, and exported tables
├── src/
│   ├── data.py           # Dataset loading and preprocessing
│   ├── model.py          # GCN architecture
│   ├── perturbations.py  # Graph and feature perturbation functions
│   ├── train.py          # Training and evaluation loop
│   └── experiments.py    # Optimizer comparison experiments
├── TASKS.md              # Work division and step-by-step roadmap
└── requirements.txt
```

## First Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## First Milestone

1. Load Cora with PyTorch Geometric.
2. Train a baseline GCN without perturbations.
3. Compare the five optimizers with fixed hyperparameters.
4. Add graph perturbations and evaluate robustness.
5. Produce final plots, tables, and project report.

