# Final Report V2 - GCN Optimizer Robustness Under Random Graph Perturbations

## 1. Abstract

This V2 report upgrades the original single-seed Cora optimizer comparison into a reproducible experimental design. It separates legacy V1 results from V2 raw outputs, defines random perturbations precisely, records environment metadata, and reports uncertainty when enough completed seeds exist. The study compares Adam, AdamW, RMSProp, AdaGrad, and SGD for a two-layer GCN on Planetoid citation networks. No result in this report should be interpreted as adversarial robustness or as a universal optimizer ranking.

## 2. Introduction and Research Question

The research question is: under a shared GCN architecture, how does optimizer choice affect node-classification performance when random graph or feature perturbations are applied? V2 scopes the answer to the dataset, model, seed set, hyperparameter protocol, and perturbation regime actually executed.

## 3. Related Work

GCNs aggregate features over graph neighborhoods, making them sensitive to both node-feature quality and graph structure. Optimizers such as Adam, AdamW, RMSProp, AdaGrad, and SGD differ in step adaptation, implicit regularization, and convergence behavior. Random perturbation studies are useful stress tests, but they are distinct from adversarial attack studies.

## 4. Experimental Design

V2 defines config-driven protocols in `configs/`. Every run records the dataset, optimizer, base seed, resolved seed, protocol, robustness setting, perturbation type, requested severity, actual perturbation count/rate, hyperparameters, metrics, timing, and environment metadata path.

## 5. Datasets and GCN Architecture

The supported datasets are Cora, CiteSeer, and PubMed from PyTorch Geometric's Planetoid interface. The model is a two-layer GCN with 16 hidden channels and dropout 0.5 in the fixed protocol. The GCN layers do not cache adjacency, which avoids leaking a clean adjacency into inference-time perturbation evaluation.

## 6. Fixed Protocol and Tuned Protocol

The fixed protocol uses the same learning rate, weight decay, hidden size, dropout, and epoch count across optimizers. This improves comparability but does not claim that every optimizer is at its best.

The tuned protocol uses validation metrics only. The default deterministic grid is learning rate `{0.001, 0.003, 0.01, 0.03}` and weight decay `{0.0, 0.0005, 0.005}`. SGD includes momentum 0.9 as a standard tuned configuration.

## 7. Random Perturbation Definitions

- Feature masking sets a requested fraction of active non-zero node-feature entries to zero.
- Edge removal removes a requested fraction of unique undirected graph connections and preserves symmetric graph representation.
- Fake edge addition inserts random non-existing undirected node pairs, with no self-loops and no duplicates.
- V1 Gaussian feature noise is retained only as legacy behavior and is labeled by standard deviation sigma.

## 8. Training-Time Versus Inference-Time Corruption

Training-time corruption trains and evaluates on perturbed inputs. Inference-time corruption trains on the clean graph and evaluates fixed weights on perturbed inputs. These settings answer different questions and are not mixed in a single ranking.

## 9. Statistical Methodology

V2 aggregation reports mean, sample standard deviation, 95% confidence interval half-width, number of seeds, clean-to-perturbed accuracy drop, and a normalized robustness AUC score across available severities. Pairwise optimizer claims should use matched seeds. The repository includes paired bootstrap intervals, Wilcoxon signed-rank tests, and Holm correction utilities; significance wording is not used unless those matched analyses support it.

## 10. Results

The current V2 pipeline writes raw rows to `results/v2/raw/` and aggregate summaries to `results/v2/aggregated/`. If only smoke rows are present, they validate the software path but do not constitute the full scientific matrix.

Completed local V2 smoke rows at the time of this report:

| dataset | optimizer | setting | perturbation | severity | epochs | test accuracy | macro F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Cora | Adam | training-time | clean | 0.00 | 3 | 0.455 | 0.452 |
| Cora | Adam | training-time | feature masking | 0.20 | 3 | 0.287 | 0.278 |
| Cora | RMSProp | training-time | clean | 0.00 | 3 | 0.341 | 0.304 |
| Cora | RMSProp | training-time | feature masking | 0.20 | 3 | 0.186 | 0.086 |

These rows are intentionally labeled as a smoke run because they use only one seed, two optimizers, two conditions, and 3 epochs. The standard Cora fixed-protocol primary matrix contains 650 planned rows; 646 remain pending after the local smoke execution.

The benchmark protocol is implemented separately from the scientific metrics. It excludes a warm-up run, repeats local measurements, and reports median, mean, standard deviation, and IQR in `results/v2/benchmarks/`. Timing is reported as local-machine dependent and is not used as a major scientific conclusion.

The tuned-protocol script writes validation-only trial rows and locked hyperparameters to `results/v2/tuned/`. Test accuracy is not used during selection.

Legacy V1 showed Adam leading the single-seed fixed Cora aggregate, with AdamW and RMSProp close in several settings. That statement is limited to V1's seed, protocol, and Gaussian-noise/structural perturbation implementation.

## 11. Cross-Dataset Validation

The V2 cross-dataset config covers Cora, CiteSeer, and PubMed with five seeds and representative 20% perturbation conditions. Any missing cross-dataset rows are pending compute, not inferred.

## 12. Discussion

The main scientific improvement is separation of protocol choices from conclusions. Fixed-protocol results are fair comparisons under shared settings; tuned-protocol results are better suited to optimizer-specific performance. Inference-time corruption measures a different robustness property from learning directly on corrupted data.

## 13. Threats to Validity / Limitations

- Full V2 matrices can be expensive on CPU.
- Timing is local-machine dependent.
- Planetoid splits are fixed and small in training nodes.
- Random perturbations are not adversarial attacks.
- Smoke runs are pipeline checks, not final evidence.
- Low SGD train/validation gap must not be interpreted as proof of stability when performance is near chance.

## 14. Reproducibility

Use:

```bash
make setup
make test
make coverage
make lint
make format-check
make smoke-v2
make benchmark-v2
make tune-v2
make aggregate-v2
make build-site
```

For the full fixed Cora matrix:

```bash
make experiment-v2-cora
make aggregate-v2
make build-site
```

## 15. References

- Kipf, T. N., and Welling, M. Semi-Supervised Classification with Graph Convolutional Networks.
- PyTorch Geometric Planetoid datasets.
- PyTorch optimizer documentation.

## 16. Appendix with Configs and Command Examples

Primary configs:

- `configs/v1_legacy_fixed_cora.yaml`
- `configs/v2_fixed_cora.yaml`
- `configs/v2_tuned_cora.yaml`
- `configs/v2_cross_dataset.yaml`
- `configs/v2_inference_robustness.yaml`

All raw V2 rows should be regenerated from scripts, not manually edited.
