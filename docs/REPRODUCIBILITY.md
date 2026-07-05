# Reproducibility Notes

## Environment

Use Python 3.11 or 3.12. Install pinned dependencies with:

```bash
make setup
```

The V2 runner writes hardware and software metadata to `results/v2/metadata/environment.json`.

Run the lightweight quality gates with:

```bash
make test
make coverage
make lint
make format-check
```

The CI recipe is included at `docs/ci/github-actions-ci.yml`. It can be installed as an active GitHub Actions workflow by placing it under `.github/workflows/` with credentials that have the GitHub `workflow` permission.

## V1 Legacy

V1 artifacts are preserved under:

- `results/v1_legacy/`
- `reports/v1_legacy/`

V1 used Cora, one seed (`42`), 200 epochs, and one fixed hyperparameter protocol. V1 feature perturbation was Gaussian feature-noise standard deviation sigma, not percentage feature corruption.

## V2 Smoke Run

```bash
make smoke-v2
```

This validates the pipeline with a bounded local run. It is not the full scientific matrix.

## Full Fixed Cora Matrix

```bash
make experiment-v2-cora
make aggregate-v2
make build-site
```

The full matrix is substantially larger than the smoke run:

- Cora
- 10 seeds
- 5 optimizers
- clean plus feature masking, edge removal, and fake edge addition
- severities 0.05, 0.10, 0.20, 0.30

Do not fill pending rows manually. Only generated raw rows should be aggregated or reported as completed.

## Runtime Benchmarking

```bash
make benchmark-v2
```

The benchmark excludes one warm-up run, repeats local timing measurements, and writes median, mean, standard deviation, and IQR to `results/v2/benchmarks/`. These timings are local CPU/device dependent.

## Tuned Protocol

```bash
make tune-v2
```

Tuning uses validation accuracy only. The script writes trial rows and locked hyperparameters to `results/v2/tuned/`; final test evaluation should use those locked hyperparameters without reselecting on test accuracy.
