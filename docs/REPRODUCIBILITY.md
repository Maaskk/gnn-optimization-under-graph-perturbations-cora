# Reproducibility Guide

## Environment

Use Python 3.11 or 3.12. Install pinned dependencies with:

```bash
make setup
```

The experiment runners write hardware and software metadata under `results/v2/metadata/`.
Each raw result row includes the Git commit, resolved seed, requested perturbation rate,
actual perturbation rate, and hardware metadata path.

## Quality Gates

```bash
make test
make lint
make format-check
```

## Isolated Smoke Run

```bash
make smoke
```

The smoke command writes only to `results/ci_smoke/`. It is for pipeline validation and
does not replace the scientific matrices.

## Main Cora Matrix

```bash
make experiment-cora
make aggregate
make diagnostics-v2
make build-site
```

The main Cora matrix contains 650 generated rows:

- Cora
- 10 seeds
- 5 optimizers
- clean graph plus feature masking, edge removal, and fake edge addition
- severities 0.05, 0.10, 0.20, 0.30
- 200 epochs per run

## Additional Validation Protocols

```bash
make experiment-cross-dataset
make experiment-tuned
make experiment-inference
make aggregate
```

Cross-dataset validation runs Cora, CiteSeer, and PubMed with five seeds and four
conditions. The tuned protocol uses validation metrics only and then locks the selected
hyperparameters before final test evaluation. Inference-time robustness trains on clean
Cora and perturbs inputs only during evaluation.

## Optimizer Diagnostics

```bash
make diagnostics-v2
```

The diagnostics command runs clean Cora training for each optimizer and seed, logs the
L2 gradient norm for every epoch, and records CPU/GPU memory summaries.

## Final Reproduction

```bash
make reproduce-final
```

This command is intentionally long: it runs tests, linting, formatting checks, the final
experimental protocols, aggregation, diagnostics, and website generation.

Do not fill missing rows manually. Only generated raw rows should be aggregated or
reported as completed.
