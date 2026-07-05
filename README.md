# Optimization of Graph Neural Networks under Random Graph Perturbations

This repository studies how optimizer choice affects a two-layer Graph Convolutional Network (GCN) under random graph and feature perturbations. The project began as a polished single-seed Cora class project and now includes a V2 reproducibility upgrade with multi-seed experiment design, explicit perturbation definitions, raw-result metadata, and static GitHub Pages reporting.

Dashboard: <https://maaskk.github.io/gnn-optimization-under-graph-perturbations-cora/>

## Research Question

For Planetoid citation networks, how do Adam, AdamW, RMSProp, AdaGrad, and SGD compare under a shared GCN architecture when random feature masking, random edge removal, and random fake-edge insertion are applied?

The project does **not** claim adversarial robustness. Perturbations are random, not optimized attacks.

## V1 Legacy Versus V2

V1 is preserved as `Legacy V1 - single seed fixed protocol`:

- dataset: Cora only
- seed: 42
- epochs: 200
- one fixed hyperparameter protocol
- Gaussian feature-noise standard deviation sigma values 0.05, 0.10, 0.20, 0.30
- directed V1 edge removal and fake-edge addition

V1 artifacts remain in their original locations for the existing dashboard and are archived under:

- `results/v1_legacy/`
- `reports/v1_legacy/`

V2 adds:

- deterministic multi-seed config files in `configs/`
- percentage-based feature masking of active non-zero feature entries
- undirected graph-consistent edge removal and fake-edge addition
- training-time and inference-time corruption protocols
- raw-result rows with resolved seeds, git commit, hyperparameters, actual perturbation counts, and environment metadata
- aggregation with seed counts, standard deviation, 95% confidence intervals, clean-to-perturbed drops, and robustness AUC scores
- matched-seed statistical utilities for paired bootstrap intervals, Wilcoxon signed-rank tests, and Holm correction

## Supported Datasets

V2 uses PyTorch Geometric Planetoid datasets:

- Cora
- CiteSeer
- PubMed

## Perturbation Definitions

- `feature_masking`: randomly sets a requested fraction of active non-zero feature entries to zero.
- `edge_removal`: removes a requested fraction of unique undirected graph connections and writes a symmetric edge representation.
- `fake_edge_addition`: inserts random previously unconnected undirected node pairs, with no self-loops or duplicates, and writes a symmetric edge representation.
- `gaussian_feature_noise_sigma`: retained only as legacy V1 behavior or optional exploratory analysis.

## Protocols

### Fixed Protocol

The fixed protocol keeps the same GCN architecture and hyperparameters for every optimizer. It favors controlled comparability, not each optimizer's best possible performance.

### Tuned Protocol

The tuned protocol uses validation metrics only. Test accuracy is not used for hyperparameter selection. The default finite grid is:

- learning rate: 0.001, 0.003, 0.01, 0.03
- weight decay: 0.0, 0.0005, 0.005
- SGD momentum: 0.9

## Training-Time Versus Inference-Time Corruption

- `training_time`: perturb graph/features, train the GCN on the perturbed input, and test on that perturbed input.
- `inference_time`: train the GCN on clean data, keep weights fixed, and evaluate perturbed inputs.

These settings answer different questions and should not be mixed in one ranking.

## Reproducibility Commands

```bash
make setup
make test
make coverage
make lint
make format-check
make smoke-v2
make benchmark-v2
make tune-v2
make experiment-v2-cora
make experiment-v2-cross-dataset
make aggregate-v2
make build-site
make reproduce-v2-standard
```

The smoke command runs a bounded local subset. It validates the pipeline but is not the full scientific matrix.

The GitHub Actions CI recipe is stored at `docs/ci/github-actions-ci.yml`. It mirrors the local quality gates and smoke experiment; installing it under `.github/workflows/` requires a GitHub token or browser session with workflow permission.

## Directory Tree

| Path | Purpose |
| --- | --- |
| `configs/` | V1/V2 experiment configurations |
| `src/gnn_robustness/` | Data loading, model, perturbations, training, V2 configs, aggregation, metadata |
| `scripts/` | V1 scripts, V2 runner, aggregation, static site data build |
| `scripts/benchmark_v2_runtime.py` | Local-machine timing protocol with warm-up and repeated measurements |
| `scripts/tune_v2_hyperparameters.py` | Validation-only finite-grid tuning before locked test evaluation |
| `results/` | V1 outputs plus `results/v2/raw/` and `results/v2/aggregated/` |
| `reports/` | V1 reports plus V2 Markdown/LaTeX report sources |
| `docs/` | Static GitHub Pages dashboard |
| `docs/ci/github-actions-ci.yml` | GitHub Actions CI recipe for tests, linting, formatting, smoke run, and site validation |
| `notebooks/` | Executable V2 reproducibility notebook |
| `tests/` | Unit and contract tests |

## Current Limitations

- Full V2 10-seed matrices can be CPU-expensive and may not be completed in one local session.
- Timing results are local-machine dependent and should not be treated as a primary scientific conclusion.
- The animated graph on the website is an illustrative sampled Cora graph unless replaced by generated V2 embedding artifacts.
- V1 conclusions are single-seed fixed-protocol observations, not universal optimizer claims.

## Key Artifacts

- `reports/final_report_v2.md`
- `reports/final_report_v2.tex`
- `notebooks/GNN_Robustness_V2_Reproducibility.ipynb`
- `configs/v2_fixed_cora.yaml`
- `results/v2/raw/`
- `results/v2/aggregated/`
- `docs/REPRODUCIBILITY.md`
