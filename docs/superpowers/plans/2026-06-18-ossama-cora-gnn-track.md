# Ossama Cora GNN Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Ossama's independent project track: clean Cora GCN optimizer comparison plus feature-noise robustness experiments.

**Architecture:** The code is split into focused Python modules under `src/gnn_robustness/`: dataset loading, model definition, optimizer factory, feature perturbations, metrics, training, plotting, and experiment orchestration. A command-line script runs the clean and feature-noise experiments and writes CSV/PNG outputs using the shared project schema.

**Tech Stack:** Python 3.12, PyTorch, PyTorch Geometric, scikit-learn, pandas, matplotlib, pytest.

---

### Task 1: Test Core Contracts

**Files:**
- Create: `tests/test_optimizers.py`
- Create: `tests/test_perturbations.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_results_schema.py`

- [ ] **Step 1: Write failing tests**

```bash
PYTHONPATH=src python -m pytest tests -q
```

Expected: FAIL because `gnn_robustness` modules do not exist yet.

- [ ] **Step 2: Implement minimal modules**

Create optimizer, perturbation, metric, and result-schema modules that satisfy the tests.

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=src python -m pytest tests -q
```

Expected: all tests pass.

### Task 2: Implement GCN Training Track

**Files:**
- Create: `src/gnn_robustness/data.py`
- Create: `src/gnn_robustness/model.py`
- Create: `src/gnn_robustness/train.py`
- Create: `src/gnn_robustness/experiments.py`
- Create: `scripts/run_ossama_track.py`

- [ ] **Step 1: Load Cora**

Use `torch_geometric.datasets.Planetoid(root="data", name="Cora", transform=NormalizeFeatures())`.

- [ ] **Step 2: Train GCN**

Use a 2-layer GCN with hidden size 16, ReLU, dropout 0.5, and cross-entropy on `train_mask`.

- [ ] **Step 3: Compare optimizers**

Run Adam, AdamW, RMSProp, AdaGrad, and SGD for clean Cora and feature-noise severities `0.05`, `0.10`, `0.20`, `0.30`. Keep AMSGrad available as a bonus optimizer via CLI flag.

- [ ] **Step 4: Save outputs**

Write `results/ossama_clean_results.csv`, `results/ossama_feature_noise_results.csv`, and figures under `results/figures/`.

### Task 3: Verify And Push

**Files:**
- Modify: `README.md`
- Create: `reports/ossama_section.md`

- [ ] **Step 1: Run tests**

```bash
PYTHONPATH=src python -m pytest tests -q
```

- [ ] **Step 2: Run a smoke experiment**

```bash
python scripts/run_ossama_track.py --epochs 2 --severities 0.05 --output-dir results/smoke
```

- [ ] **Step 3: Run the project experiment**

```bash
python scripts/run_ossama_track.py --epochs 200
```

- [ ] **Step 4: Commit and push**

```bash
git add .
git commit -m "feat: implement Ossama Cora optimizer track"
git push origin ossama/baseline
```

